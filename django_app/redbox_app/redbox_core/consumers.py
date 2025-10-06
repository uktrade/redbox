import json
import logging
from asyncio import CancelledError
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar
from uuid import UUID

from asgiref.sync import sync_to_async
from botocore.exceptions import ClientError
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from django.contrib.auth import get_user_model
from django.forms.models import model_to_dict
from django.utils import timezone
from langchain_core.documents import Document
from pydantic import ValidationError
from websockets import ConnectionClosedError, WebSocketClientProtocol

from redbox import Redbox
from redbox.models.chain import (
    AISettings,
    ChainChatMessage,
    MultiAgentPlan,
    RedboxQuery,
    RedboxState,
    RequestMetadata,
    Source,
    get_plan_fix_prompts,
    metadata_reducer,
)
from redbox.models.chain import Citation as AICitation
from redbox.models.graph import RedboxActivityEvent
from redbox.models.settings import get_settings
from redbox_app.redbox_core import error_messages
from redbox_app.redbox_core.models import (
    ActivityEvent,
    AgentPlan,
    Chat,
    ChatLLMBackend,
    ChatMessage,
    ChatMessageTokenUse,
    Citation,
    File,
    MonitorSearchRoute,
)
from redbox_app.redbox_core.models import AISettings as AISettingsModel
from redbox_app.redbox_core.services import message as message_service

convert_american_to_british_spelling = sync_to_async(
    message_service.convert_american_to_british_spelling, thread_sensitive=True
)

User = get_user_model()
OptFileSeq = Sequence[File] | None
logger = logging.getLogger(__name__)
logger.info("WEBSOCKET_SCHEME is: %s", settings.WEBSOCKET_SCHEME)


def parse_page_number(obj: int | list[int] | None) -> list[int]:
    if isinstance(obj, int):
        return [obj]
    elif isinstance(obj, list) and len(obj) > 0 and all(isinstance(item, int) for item in obj):
        return obj
    elif obj is None:
        return []
    msg = "expected, int | list[int] | None got %s"
    raise ValueError(msg, type(obj))


def escape_curly_brackets(text: str):
    return text.replace("{", "{{").replace("}", "}}")


@sync_to_async
def get_latest_complete_file(ref: str) -> File:
    return File.objects.filter(original_file=ref, status=File.Status.complete).order_by("-created_at").first()


class ChatConsumer(AsyncWebsocketConsumer):
    full_reply: ClassVar = []
    converted_reply: ClassVar = []
    citations: ClassVar[list[tuple[File, AICitation]]] = []
    activities: ClassVar[list[RedboxActivityEvent]] = []
    route = None
    metadata: RequestMetadata = RequestMetadata()
    redbox = Redbox(env=get_settings(), debug=True)
    chat_message = None  # incrementally updating the chat stream

    async def receive(self, text_data=None, bytes_data=None):
        """Receive & respond to message from browser websocket."""
        self.full_reply = []
        self.converted_reply = []
        self.citations = []
        self.external_citations = []
        self.route = None
        self.activities = []
        self.metadata = RequestMetadata()
        self.chat_message = None
        self.final_state = None

        data = json.loads(text_data or bytes_data)
        logger.debug("received %s from browser", data)
        user_message_text: str = data.get("message", "")
        selected_file_uuids: Sequence[UUID] = [UUID(u) for u in data.get("selectedFiles", [])]
        activities: Sequence[str] = data.get("activities", [])
        user: User = self.scope.get("user")

        user_ai_settings = await AISettingsModel.objects.aget(label=user.ai_settings_id if user else "Claude 3.7")

        chat_backend = await ChatLLMBackend.objects.aget(id=data.get("llm", user_ai_settings.chat_backend_id))
        temperature = data.get("temperature", user_ai_settings.temperature)

        if session_id := data.get("sessionId"):
            session = await Chat.objects.aget(id=session_id)
            session.chat_backend = chat_backend
            session.temperature = temperature
            logger.info("updating session: chat_backend=%s temperature=%s", chat_backend, temperature)
            await session.asave()
        else:
            logger.info("creating session: chat_backend=%s temperature=%s", chat_backend, temperature)
            session = await Chat.objects.acreate(
                name=user_message_text[: settings.CHAT_TITLE_LENGTH],
                user=user,
                chat_backend=chat_backend,
                temperature=temperature,
            )

        # save user message
        permitted_files = File.objects.filter(user=user, status=File.Status.complete)
        selected_files = permitted_files.filter(id__in=selected_file_uuids)
        await self.save_user_message(session, user_message_text, selected_files=selected_files, activities=activities)

        self.chat_message = await self.create_ai_message(session)

        await self.llm_conversation(selected_files, session, user, user_message_text, permitted_files)

        if (self.final_state) and (self.final_state.agent_plans):
            await self.agent_plan_save(session)

        # save user, ai and intermediary graph outputs if 'search' route is invoked
        if self.route == "search":
            await self.monitor_search_route(session, user_message_text)

        await self.close()

    async def llm_conversation(
        self, selected_files: Sequence[File], session: Chat, user: User, title: str, permitted_files: Sequence[File]
    ) -> None:
        """Initiate & close websocket conversation with the core-api message endpoint."""
        await self.send_to_client("session-id", session.id)

        session_messages = ChatMessage.objects.filter(chat=session).order_by("created_at")
        message_history: Sequence[Mapping[str, str]] = [message async for message in session_messages]
        question = message_history[-2].text
        user_feedback = ""
        plan_message_size = 3
        agent_plans = None

        plan_prefix, _ = get_plan_fix_prompts()
        if (len(message_history) > plan_message_size) and (
            any(n in message_history[-plan_message_size].text for n in plan_prefix)
        ):

            @sync_to_async
            def get_agent_plan(session: Chat):
                return (
                    AgentPlan.objects.filter(chat__id=session.id)
                    .order_by("-created_at")
                    .values_list("agent_plans")
                    .first()
                )

            try:
                plan = await get_agent_plan(session)
            except AgentPlan.DoesNotExist as e:
                logger.debug("Cannot find object in db %s", e)
            if plan:
                try:
                    agent_plans = MultiAgentPlan.model_validate_json(plan[0])
                    question = message_history[-4].text
                    user_feedback = message_history[-2].text
                    logger.debug("here is the plan: %s", plan[0])
                    logger.debug("here is user feedback %s", user_feedback)
                except ValidationError as e:
                    logger.debug("cannot parse into plan object %s", plan[0])
                    logger.exception("Error from converting plan.", exc_info=e)

        ai_settings = await self.get_ai_settings(session)
        state = RedboxState(
            request=RedboxQuery(
                question=question,
                s3_keys=[f.unique_name for f in selected_files],
                user_uuid=user.id,
                chat_history=[
                    ChainChatMessage(
                        role=message.role,
                        text=escape_curly_brackets(message.text),
                    )
                    for message in message_history[:-2]
                    if message.text
                ],
                ai_settings=ai_settings,
                permitted_s3_keys=[f.unique_name async for f in permitted_files],
            ),
            user_feedback=user_feedback,
            agent_plans=agent_plans,
        )

        try:
            self.final_state = await self.redbox.run(
                state,
                response_tokens_callback=self.handle_text,
                route_name_callback=self.handle_route,
                documents_callback=self.handle_documents,
                citations_callback=self.handle_citations,
                metadata_tokens_callback=self.handle_metadata,
                activity_event_callback=self.handle_activity,
            )
            await self.update_ai_message()
            if len(self.full_reply) == 0 or self.chat_message.text == "":
                logger.exception("LLM Error - Blank Response")
            await self.send_to_client(
                "end", {"message_id": self.chat_message.id, "title": title, "session_id": session.id}
            )

        except ClientError as e:
            logger.exception("Rate limit error", exc_info=e)
            await self.send_to_client("error", error_messages.RATE_LIMITED)
        except (TimeoutError, ConnectionClosedError, CancelledError) as e:
            logger.exception("Error from core.", exc_info=e)
            await self.send_to_client("error", error_messages.CORE_ERROR_MESSAGE)
        except Exception as e:
            logger.exception("General error.", exc_info=e)
            await self.send_to_client("error", error_messages.CORE_ERROR_MESSAGE)

    async def send_to_client(self, message_type: str, data: str | Mapping[str, Any] | None = None) -> None:
        message = {"type": message_type, "data": data}
        logger.debug("sending %s to browser", message)
        await self.send(json.dumps(message, default=str))

    @staticmethod
    async def send_to_server(websocket: WebSocketClientProtocol, data: Mapping[str, Any]) -> None:
        logger.debug("sending %s to core-api", data)
        return await websocket.send(json.dumps(data, default=str))

    @database_sync_to_async
    def save_user_message(
        self,
        session: Chat,
        user_message_text: str,
        selected_files: Sequence[File] | None = None,
        activities: Sequence[str] | None = None,
    ) -> ChatMessage:
        chat_message = ChatMessage(
            chat=session,
            text=user_message_text,
            role=ChatMessage.Role.user,
            route=self.route,
        )
        chat_message.save()
        if selected_files:
            chat_message.selected_files.set(selected_files)

        # Save user activities
        for message in activities or []:
            ActivityEvent.objects.create(chat_message=chat_message, message=message)

        chat_message.log()
        return chat_message

    @database_sync_to_async
    def create_ai_message(self, session: Chat) -> ChatMessage:
        chat_message = ChatMessage(
            chat=session,
            text="",
            role=ChatMessage.Role.ai,
            route=self.route,
        )
        chat_message.save()
        return chat_message

    @database_sync_to_async
    def update_ai_message(self) -> None:
        if not self.chat_message:
            logger.error("No chat message to update")
            return

        # convert text to British English
        converted_text = "".join(self.converted_reply)
        logger.debug("Saving converted text: %s", converted_text[:50])
        self.chat_message.text = converted_text
        self.chat_message.route = self.route
        self.chat_message.save()
        # Important - clears existing citations and related objects to avoid duplicates
        Citation.objects.filter(chat_message=self.chat_message).delete()
        ChatMessageTokenUse.objects.filter(chat_message=self.chat_message).delete()
        ActivityEvent.objects.filter(chat_message=self.chat_message).delete()

        # Save citations
        for file, ai_citation in self.citations:
            for citation_source in ai_citation.sources:
                if file:
                    file.last_referenced = timezone.now()
                    file.save()
                    Citation.objects.create(
                        chat_message=self.chat_message,
                        text_in_answer=ai_citation.text_in_answer,
                        file=file,
                        text=citation_source.highlighted_text_in_source,
                        page_numbers=citation_source.page_numbers,
                        source=Citation.Origin.USER_UPLOADED_DOCUMENT,
                        citation_name=citation_source.ref_id,
                    )
                else:
                    Citation.objects.create(
                        chat_message=self.chat_message,
                        text_in_answer=ai_citation.text_in_answer,
                        url=citation_source.source,
                        text=citation_source.highlighted_text_in_source,
                        page_numbers=citation_source.page_numbers,
                        source=Citation.Origin.try_parse(citation_source.source_type),
                        citation_name=citation_source.ref_id,
                    )

        if self.metadata:
            for model, token_count in self.metadata.input_tokens.items():
                ChatMessageTokenUse.objects.create(
                    chat_message=self.chat_message,
                    use_type=ChatMessageTokenUse.UseType.INPUT,
                    model_name=model,
                    token_count=token_count,
                )
            for model, token_count in self.metadata.output_tokens.items():
                ChatMessageTokenUse.objects.create(
                    chat_message=self.chat_message,
                    use_type=ChatMessageTokenUse.UseType.OUTPUT,
                    model_name=model,
                    token_count=token_count,
                )

        if self.activities:
            for activity in self.activities:
                ActivityEvent.objects.create(chat_message=self.chat_message, message=activity.message)

        self.chat_message.log()

    @database_sync_to_async
    def agent_plan_save(self, session: Chat) -> AgentPlan:
        logger.info("Saving agent plans")
        agent_plan = AgentPlan(chat=session, agent_plans=self.final_state.agent_plans.model_dump_json())
        agent_plan.save()
        return agent_plan

    @database_sync_to_async
    def monitor_search_route(
        self,
        session: Chat,
        user_message_text: str,
    ) -> MonitorSearchRoute:
        user_rephrased_text = self.final_state.messages[0].content
        similarity_scores = {}
        if self.final_state.documents is not None:
            for i, group in enumerate(self.final_state.documents.groups.values()):
                for val in group.values():
                    similarity_scores[i] = {
                        "uuid": val.metadata["uuid"],
                        "score": val.metadata["score"],
                    }

        monitor_search = MonitorSearchRoute(
            chat=session,
            user_text=user_message_text,
            user_text_rephrased=user_rephrased_text,
            route=self.route,
            chunk_similarity_scores=similarity_scores,
            ai_text="".join(self.converted_reply),
        )
        monitor_search.save()
        return monitor_search

    @database_sync_to_async
    def get_ai_settings(self, chat: Chat) -> AISettings:
        ai_settings = model_to_dict(chat.user.ai_settings, exclude=["label", "chat_backend"])
        ai_settings["chat_backend"] = model_to_dict(chat.chat_backend)

        # we remove null values so that AISettings can populate them with defaults
        ai_settings = {k: v for k, v in ai_settings.items() if v not in (None, "")}
        return AISettings.model_validate(ai_settings)

    async def handle_text(self, response: str) -> str:
        """Handle text chunks and British spelling conversion before sending to client."""
        logger.debug("Received text chunk: %s", response)
        try:
            converted_chunk = (
                convert_american_to_british_spelling(response) if self.scope.get("user").uk_or_us_english else response
            )
            logger.debug("converted text chunk: %s -> %s", response[:50], converted_chunk[:50])
        except Exception as e:
            logger.exception("conversion failed ", exc_info=e)
            converted_chunk = response  # use unconverted text

        # store both original and converted chunks
        self.full_reply.append(response)
        self.converted_reply.append(converted_chunk)

        # send converted text to client
        await self.send_to_client("text", converted_chunk)
        await self.update_ai_message()
        return converted_chunk

    async def handle_route(self, response: str) -> str:
        await self.send_to_client("route", response)
        self.route = response
        await self.update_ai_message()
        return response

    async def handle_metadata(self, response: dict):
        self.metadata = metadata_reducer(self.metadata, RequestMetadata.model_validate(response))
        await self.update_ai_message()

    async def handle_activity(self, response: dict):
        await self.send_to_client("activity", response.message)
        self.activities.append(RedboxActivityEvent.model_validate(response))
        await self.update_ai_message()

    async def handle_documents(self, response: list[Document]):
        """
        Map documents used to create answer to AICitations for storing as citations
        """
        sources_by_resource_ref: dict[str, Document] = defaultdict(list)
        for document in response:
            ref = document.metadata.get("uri")
            sources_by_resource_ref[ref].append(document)

        for ref, sources in sources_by_resource_ref.items():
            try:
                # Use the async database query function
                file = await get_latest_complete_file(ref)
                if file:
                    payload = {"url": str(file.url), "file_name": file.file_name, "text_in_answer": ""}
                else:
                    # If no file with Status.complete is found, handle it as None
                    payload = {"url": ref, "file_name": None, "text_in_answer": ""}

                response_sources = [
                    Source(
                        source=str(file.url if file else ref),
                        source_type=Citation.Origin.USER_UPLOADED_DOCUMENT,
                        document_name=file.file_name if file else ref.split("/")[-1],
                        highlighted_text_in_source=cited_chunk.page_content,
                        page_numbers=parse_page_number(cited_chunk.metadata.get("page_number")),
                    )
                    for cited_chunk in sources
                ]
            except File.DoesNotExist:
                file = None
                payload = {"url": ref, "file_name": None, "text_in_answer": ""}
                response_sources = [
                    Source(
                        source=cited_chunk.metadata["uri"],
                        source_type=cited_chunk.metadata["creator_type"],
                        document_name=cited_chunk.metadata["uri"].split("/")[-1],
                        highlighted_text_in_source=cited_chunk.page_content,
                        page_numbers=parse_page_number(cited_chunk.metadata.get("page_number")),
                    )
                    for cited_chunk in sources
                ]

            await self.send_to_client("source", payload)
            self.citations.append((file, AICitation(text_in_answer="", sources=response_sources)))
            await self.update_ai_message()

    async def handle_citations(self, citations: list[AICitation]):
        """
        Map AICitations used to create answer to AICitations for storing as citations. The link to user files
        must be populated
        """
        for c in citations:
            for s in c.sources:
                try:
                    # Use the async database query function
                    file = await get_latest_complete_file(s.source)
                    if file:
                        payload = {
                            "url": str(file.url),
                            "file_name": file.file_name,
                            "text_in_answer": convert_american_to_british_spelling(c.text_in_answer)
                            if self.scope.get("user").uk_or_us_english
                            else c.text_in_answer,
                            "citation_name": s.ref_id,
                        }
                    else:
                        # if source is empty, attempt to filter by document name
                        file = await get_latest_complete_file(s.document_name)
                        if file:
                            payload = {
                                "url": str(file.url),
                                "file_name": file.file_name,
                                "text_in_answer": convert_american_to_british_spelling(c.text_in_answer)
                                if self.scope.get("user").uk_or_us_english
                                else c.text_in_answer,
                                "citation_name": s.ref_id,
                            }
                        else:
                            # If no file with Status.complete is found, handle it as None
                            payload = {
                                "url": s.source,
                                "file_name": s.source,
                                "text_in_answer": convert_american_to_british_spelling(c.text_in_answer)
                                if self.scope.get("user").uk_or_us_english
                                else c.text_in_answer,
                                "citation_name": s.ref_id,
                            }
                except File.DoesNotExist:
                    file = None
                    text_in_answer = c.text_in_answer or ""
                    payload = {
                        "url": s.source,
                        "file_name": s.source,
                        "text_in_answer": convert_american_to_british_spelling(text_in_answer)
                        if self.scope.get("user").uk_or_us_english
                        else text_in_answer,
                        "citation_name": s.ref_id,
                    }

                await self.send_to_client("source", payload)

                text_in_answer = (
                    await convert_american_to_british_spelling(c.text_in_answer)
                    if self.scope.get("user").uk_or_us_english
                    else c.text_in_answer
                )

                self.citations.append(
                    (
                        file,
                        AICitation(
                            text_in_answer=text_in_answer,
                            sources=[s],
                        ),
                    )
                )
            await self.update_ai_message()

    async def handle_activity_event(self, event: RedboxActivityEvent):
        logger.info("ACTIVITY: %s", event.message)
