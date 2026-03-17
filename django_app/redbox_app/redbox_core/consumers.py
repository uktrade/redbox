import asyncio
import json
import logging
import re
from asyncio import CancelledError
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar
from uuid import UUID

import aiohttp
import uwotm8.convert as uwm8
from asgiref.sync import sync_to_async
from botocore.exceptions import ClientError
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db.models import Q
from django.forms.models import model_to_dict
from django.utils import timezone
from langchain_core.documents import Document
from pydantic import ValidationError
from waffle import flag_is_active
from websockets import ConnectionClosedError, WebSocketClientProtocol

from redbox import Redbox
from redbox.graph.agents.configs import agent_configs
from redbox.graph.nodes.tools import get_datahub_mcp_tools
from redbox.models.chain import (
    AISettings,
    ChainChatMessage,
    RedboxQuery,
    RedboxState,
    RequestMetadata,
    Source,
    configure_agent_task_plan,
    get_plan_fix_prompts,
    metadata_reducer,
)
from redbox.models.chain import Citation as AICitation
from redbox.models.graph import RedboxActivityEvent
from redbox.models.settings import ChatLLMBackend, get_settings
from redbox_app.redbox_core import error_messages, flags
from redbox_app.redbox_core.models import (
    ActivityEvent,
    AgentPlan,
    AgentTool,
    Chat,
    ChatMessage,
    ChatMessageTokenUse,
    Citation,
    File,
    FileTeamMembership,
    FileTool,
    MonitorSearchRoute,
    MonitorWebSearchResults,
    Tool,
    UserTeamMembership,
)
from redbox_app.redbox_core.models import Agent as AgentModel
from redbox_app.redbox_core.models import AISettings as AISettingsModel
from redbox_app.redbox_core.models import ChatLLMBackend as ChatLLMBackendModel

# Temporary condition before next uwotm8 release: monkey patch CONVERSION_IGNORE_LIST
uwm8.CONVERSION_IGNORE_LIST = uwm8.CONVERSION_IGNORE_LIST | {"filters": "philtres", "connection": "connexion"}
User = get_user_model()
OptFileSeq = Sequence[File] | None
logger = logging.getLogger(__name__)
logger.info("WEBSOCKET_SCHEME is: %s", settings.WEBSOCKET_SCHEME)


async def wait_for_health(
    url: str,
    headers: dict,
    success_codes: list[int],
    wait_seconds: float = 60.0,
    interval: float = 2.0,
    request_timeout: float = 5.0,
) -> bool:
    """
    Polls URL for a response

    Parameters:
        url: str - the endpoint to poll
        headers: dict - headers to embed in the endpoint call
        success_codes: list[int] - successful response codes ie. [200]
        wait_seconds: float - the deadline for polling
        interval: float - the interval for polling
        request_timeout: float - the timeout for the request

    Outputs:
        bool - True if successful response, False otherwise
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + wait_seconds
    attempt = 0

    while loop.time() < deadline:
        attempt += 1
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=request_timeout),
                    headers=headers,
                ) as response,
            ):
                if response.status in success_codes:
                    logger.info("Health check passed after %d attempt(s)", attempt)
                    return True
                logger.debug("Health check attempt %d: status %d", attempt, response.status)
        except (
            aiohttp.ClientConnectionError,  # connection refused, DNS failure
            aiohttp.ClientResponseError,  # bad HTTP response
            aiohttp.ServerTimeoutError,  # per-request timeout
            TimeoutError,  # underlying socket timeout
        ) as e:
            logger.debug("Health check attempt %d failed: %s", attempt, e)

        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(interval, remaining))

    logger.warning("Health check timed out after %d attempt(s): %s", attempt, url)
    return False


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


@database_sync_to_async
def get_latest_complete_file(ref: str) -> File:
    return File.objects.filter(original_file__endswith=ref, status=File.Status.complete).order_by("-created_at").first()


@database_sync_to_async
def get_all_agents():
    return tuple(AgentModel.objects.select_related("llm_backend").all())


class ChatConsumer(AsyncWebsocketConsumer):
    route = None
    metadata: RequestMetadata = RequestMetadata()
    env = get_settings()
    debug = not env.is_prod
    redbox = None
    chat_message = None  # incrementally updating the chat stream

    _mcp_monitor_tasks: ClassVar[dict[str, asyncio.Task]] = {}
    _monitor_task_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _tools_loaded: ClassVar[dict[str, bool]] = {}
    _sso_token: str | None = None

    async def get_file_cached(self, ref):
        if ref not in self._file_cache:
            self._file_cache[ref] = await get_latest_complete_file(ref)
        return self._file_cache[ref]

    async def _init_session(self, data, user, user_message_text, chat_backend, temperature):
        """Create or update a chat session."""
        if session_id := data.get("sessionId"):
            session = await Chat.objects.aget(id=session_id)
            session.chat_backend = chat_backend
            session.temperature = temperature
            logger.info("updating session: chat_backend=%s temperature=%s", chat_backend, temperature)
            await session.asave()
            if data.get("selectedFiles", []):
                # get previous selected files from db
                latest_message = (
                    await ChatMessage.objects.filter(chat=session, role="user").order_by("-created_at").afirst()
                )
                if latest_message:
                    latest_files = latest_message.selected_files.all()
                    previous_selected_files = [file async for file in latest_files]

            else:
                previous_selected_files = []
        else:
            logger.info("creating session: chat_backend=%s temperature=%s", chat_backend, temperature)
            session = await Chat.objects.acreate(
                name=user_message_text[: settings.CHAT_TITLE_LENGTH],
                user=user,
                chat_backend=chat_backend,
                temperature=temperature,
            )
            previous_selected_files = []
        return session, previous_selected_files

    async def receive(self, text_data=None, bytes_data=None):  # noqa: C901, PLR0915
        """Receive & respond to message from browser websocket."""
        # if user is unauthenticated, close connection
        user = getattr(self, "user", None) or self.scope.get("user")
        if (not user) or (not user.is_authenticated):
            await self.close(code=4001)
            return
        self._file_cache = {}
        self.full_reply = []
        self.converted_reply = []
        self.citations = []
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
        selected_tool_id: str | None = data.get("selectedTool")
        user: User = self.scope["user"]

        user_ai_settings = await AISettingsModel.objects.aget(label=user.ai_settings_id if user else "Claude 3.7")
        user_team_ids = await database_sync_to_async(
            lambda: list(UserTeamMembership.objects.filter(user=user).values_list("team_id", flat=True))
        )()

        chat_backend = await ChatLLMBackendModel.objects.aget(id=data.get("llm", user_ai_settings.chat_backend_id))
        temperature = data.get("temperature", user_ai_settings.temperature)

        session, previous_selected_files = await self._init_session(
            data, user, user_message_text, chat_backend, temperature
        )

        # save user message
        cache_key = f"user:{user.id}:permitted_files"

        permitted_files = await cache.aget(cache_key)
        if permitted_files is None:
            qs = (
                File.objects.filter(
                    Q(user=user, status=File.Status.complete)
                    | Q(
                        team_associations__team_id__in=user_team_ids,
                        team_associations__visibility=FileTeamMembership.Visibility.TEAM,
                        status=File.Status.complete,
                    )
                )
                .distinct()
                .only("id", "original_file", "original_file_name", "status")
            )
            permitted_files = await database_sync_to_async(list)(qs)
            await cache.aset(cache_key, permitted_files, 30)

        selected_files = [f for f in permitted_files if f.id in selected_file_uuids]

        if not session.name and selected_files:
            first_file = selected_files[0]
            session_name = await database_sync_to_async(
                lambda fid: File.objects.filter(id=fid).values_list("original_file_name", flat=True).first()
            )(first_file.id)
            session.name = (session_name or "")[: settings.CHAT_TITLE_LENGTH]
            await session.asave()

        tool_obj = None
        selected_agent_names = []
        knowledge_files = []
        if selected_tool_id:
            try:
                tool_obj = await database_sync_to_async(Tool.objects.get)(id=selected_tool_id)
                session.tool = tool_obj
                await session.asave()
                selected_agent_names = await database_sync_to_async(
                    lambda t: list(AgentTool.objects.filter(tool=t).values_list("agent__name", flat=True))
                )(tool_obj)
                knowledge_files = await database_sync_to_async(lambda t: list(t.get_files(FileTool.FileType.ADMIN)))(
                    tool_obj
                )
            except Tool.DoesNotExist:
                logger.warning("Selected tool '%s' not found", selected_tool_id)

        user_chat_message = await self.save_user_message(
            session, user_message_text, selected_files=selected_files, activities=activities, tool=tool_obj
        )

        self.chat_message = await self.create_ai_message(session)

        await self.llm_conversation(
            selected_files,
            session,
            user,
            user_message_text,
            permitted_files,
            previous_selected_files,
            selected_agent_names=selected_agent_names,
            knowledge_files=knowledge_files,
        )

        if (self.final_state) and (self.final_state.agent_plans):
            await self.agent_plan_save(session)

        # save user, ai and intermediary graph outputs if 'search' route is invoked
        if self.route == "search":
            await self.monitor_search_route(session, user_message_text)

        # save web search query and all web results from web search related agents
        if (self.final_state) and (self.final_state.agents_results):
            user_question = self.final_state.request.question
            web_search_results_urls = []
            web_search_api_counter = 0
            for task_id in self.final_state.agents_results:
                source_type = re.search(
                    "<SourceType>(.*?)</SourceType>", self.final_state.agents_results[task_id].content
                )
                if source_type and source_type.group(1) == Citation.Origin.WEB_SEARCH:
                    web_search_results_urls += re.findall(
                        "<Source>(.*?)</Source>", self.final_state.agents_results[task_id].content
                    )
                    web_search_api_counter += 1

            if web_search_results_urls:
                await self.monitor_web_search_results(
                    user_chat_message,
                    user_question,
                    web_search_results_urls,
                    web_search_api_counter,
                    selected_files=selected_files,
                )

        await self.close()

    async def _load_agent_plan(self, session: Chat, message_history: Sequence[Mapping[str, str]]):
        """Try to load and parse an existing agent plan if present."""
        plan_message_size = 3
        plan_prefix, _ = get_plan_fix_prompts()

        if (len(message_history) > plan_message_size) and (
            any(n in message_history[-plan_message_size].text for n in plan_prefix)
        ):

            @database_sync_to_async
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
                return None, "", message_history[-2].text

            if plan:
                try:
                    agent_options = {agent: agent for agent in agent_configs}
                    _, configured_agent_plan = configure_agent_task_plan(agent_options)
                    agent_plans = configured_agent_plan.model_validate_json(plan[0])
                    question = message_history[-4].text
                    user_feedback = message_history[-2].text
                    logger.debug("here is the plan: %s", plan[0])
                    logger.debug("here is user feedback %s", user_feedback)
                    return agent_plans, question, user_feedback  # noqa: TRY300
                except ValidationError as e:
                    logger.debug("cannot parse into plan object %s", plan[0])
                    logger.exception("Error from converting plan.", exc_info=e)
        return None, message_history[-2].text, ""  # message_history[-2].text extracts the current user query

    @database_sync_to_async
    def _files_to_s3_keys(self, files: Sequence[File]) -> list[str]:
        if not files:
            return []
        ids = [f.id for f in files]
        return list(File.objects.filter(id__in=ids).values_list("original_file", flat=True))

    def _extract_sso_token(self) -> str | None:
        try:
            session = self.scope.get("session")
            return session["_authbroker_token"]["access_token"]
        except (KeyError, TypeError):
            return None

    async def llm_conversation(
        self,
        selected_files: Sequence[File],
        session: Chat,
        user: User,
        title: str,
        permitted_files: Sequence[File],
        previous_selected_files: Sequence[File],
        knowledge_files: Sequence[File],
        selected_agent_names: list[str] | None = None,
    ) -> None:
        """Initiate & close websocket conversation with the core-api message endpoint."""
        await self.send_to_client("session-id", session.id)
        session_messages = ChatMessage.objects.filter(chat=session).order_by("created_at")
        message_history: Sequence[Mapping[str, str]] = [message async for message in session_messages]
        question = message_history[-2].text
        user_feedback = ""
        agent_plans = None

        agent_plans, question, user_feedback = await self._load_agent_plan(session, message_history)

        ai_settings = await self.get_ai_settings(session)
        sso_access_token = self._extract_sso_token()

        if selected_agent_names:
            ai_settings = ai_settings.model_copy(
                update={
                    "worker_agents": [agent for agent in agent_configs.values() if agent.name in selected_agent_names]
                }
            )

        if sso_access_token:
            self.redbox.agent_configs["Datahub_Agent"].tools = get_datahub_mcp_tools(sso_access_token=sso_access_token)

        state = RedboxState(
            request=RedboxQuery(
                question=question,
                s3_keys=await self._files_to_s3_keys(selected_files),
                user_uuid=user.id,
                chat_history=[
                    ChainChatMessage(role=m.role, text=escape_curly_brackets(m.text))
                    for m in message_history[:-2]
                    if m.text
                ],
                ai_settings=ai_settings,
                permitted_s3_keys=await self._files_to_s3_keys(permitted_files),
                previous_s3_keys=await self._files_to_s3_keys(previous_selected_files),
                knowledge_base_s3_keys=await self._files_to_s3_keys(knowledge_files) if knowledge_files else [],
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
                logger.exception("consumers: LLM Error - Blank Response")
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
        tool: Tool | None = None,
    ) -> ChatMessage:
        chat_message = ChatMessage(
            chat=session,
            text=user_message_text,
            role=ChatMessage.Role.user,
            route=self.route,
        )
        chat_message.tool = tool
        chat_message.save()
        if selected_files:
            chat_message.selected_files.set(selected_files)

        # Save user activities
        ActivityEvent.objects.bulk_create(
            [ActivityEvent(chat_message=chat_message, message=m) for m in activities or []]
        )

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
    def update_ai_message(self) -> None:  # noqa: C901
        if not self.chat_message:
            logger.error("No chat message to update")
            return

        self.chat_message.text = "".join(self.converted_reply)
        self.chat_message.route = self.route
        self.chat_message.save()

        # Important - clears existing citations and related objects to avoid duplicates
        Citation.objects.filter(chat_message=self.chat_message).delete()
        ChatMessageTokenUse.objects.filter(chat_message=self.chat_message).delete()
        ActivityEvent.objects.filter(chat_message=self.chat_message).delete()

        referenced_file_pks = {file.pk for file, _ in self.citations if file}
        if referenced_file_pks:
            File.objects.filter(pk__in=referenced_file_pks).update(last_referenced=timezone.now())

        # Save citations
        citation_objects = []
        token_objects = []
        activity_objects = []

        for file, ai_citation in self.citations:
            for src in ai_citation.sources:
                citation_objects.append(
                    Citation(
                        chat_message=self.chat_message,
                        file=file,
                        url=None if file else src.source,
                        text=src.highlighted_text_in_source,
                        page_numbers=src.page_numbers,
                        source=Citation.Origin.USER_UPLOADED_DOCUMENT
                        if file
                        else Citation.Origin.try_parse(src.source_type),
                        citation_name=src.ref_id,
                    )
                )

        for model, token_count in self.metadata.input_tokens.items():
            token_objects.append(
                ChatMessageTokenUse(
                    chat_message=self.chat_message,
                    use_type=ChatMessageTokenUse.UseType.INPUT,
                    model_name=model,
                    token_count=token_count,
                )
            )

        for model, token_count in self.metadata.output_tokens.items():
            token_objects.append(
                ChatMessageTokenUse(
                    chat_message=self.chat_message,
                    use_type=ChatMessageTokenUse.UseType.OUTPUT,
                    model_name=model,
                    token_count=token_count,
                )
            )

        for activity in self.activities:
            activity_objects.append(
                ActivityEvent(
                    chat_message=self.chat_message,
                    message=activity.message,
                )
            )

        if citation_objects:
            Citation.objects.bulk_create(citation_objects)

        if token_objects:
            ChatMessageTokenUse.objects.bulk_create(token_objects)

        if activity_objects:
            ActivityEvent.objects.bulk_create(activity_objects)

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
    def monitor_web_search_results(
        self,
        message: ChatMessage,
        user_message_text: str,
        web_search_urls: list,
        web_search_api_count: int,
        selected_files: Sequence[File] | None = None,
    ) -> MonitorWebSearchResults:
        logger.info("Saving web search urls")
        monitor_web_search = MonitorWebSearchResults(
            chat_message=message,
            user_text=user_message_text,
            web_search_urls=str(web_search_urls),
            web_search_api_count=web_search_api_count,
        )
        monitor_web_search.save()
        if selected_files:
            monitor_web_search.selected_files.set(selected_files)
        return monitor_web_search

    @database_sync_to_async
    def get_ai_settings(self, chat: Chat) -> AISettings:
        ai_settings = model_to_dict(chat.user.ai_settings, exclude=["label", "chat_backend"])
        ai_settings["chat_backend"] = model_to_dict(chat.chat_backend)

        # we remove null values so that AISettings can populate them with defaults
        ai_settings = {k: v for k, v in ai_settings.items() if v not in (None, "")}
        ai_settings = AISettings.model_validate(ai_settings)
        return ai_settings.model_copy(
            update={"worker_agents": [agent for agent in agent_configs.values() if agent.default_agent]}
        )

    async def connect(self):
        self.user = self.scope["user"]

        await self.accept()

        # if user is unauthenticated, send auth_required error message
        if not self.user.is_authenticated:
            await self.send_to_client("error", error_messages.AUTH_REQUIRED)
            return

        mcp_agents = {
            "Datahub_Agent": f"{ChatConsumer.env.datahub_mcp.url.removesuffix('/mcp/')}/health",
        }

        ChatConsumer._sso_token = self._extract_sso_token()

        if ChatConsumer.redbox is None:
            await self._init_redbox(mcp_agents=mcp_agents)

        # setup background task to monitor health of MCP servers
        async with ChatConsumer._monitor_task_lock:
            for agent_name, health_url in mcp_agents.items():
                task = ChatConsumer._mcp_monitor_tasks.get(agent_name)
                if task is None or task.done():
                    ChatConsumer._mcp_monitor_tasks[agent_name] = asyncio.create_task(
                        ChatConsumer._mcp_monitor_loop(
                            agent_name=agent_name,
                            mcp_url=health_url,
                            get_token=lambda: ChatConsumer._sso_token,
                        )
                    )

        self.uk_english = await database_sync_to_async(lambda u: getattr(u, "uk_or_us_english", False))(self.user)

    async def _init_redbox(self, mcp_agents: dict[str, str]):
        """
        Initialise or reinitialise Redbox

        Inputs:
            mcp_agents: dict[str, str] - key-value pairs of MCP agent names and their health endpoint URL
        """
        agents = await get_all_agents()
        sso_access_token = self._extract_sso_token()

        for agent in agents:
            if agent.name in agent_configs:
                if agent.llm_backend:
                    agent_configs[agent.name].llm_backend = ChatLLMBackend(
                        name=agent.llm_backend.name,
                        provider=agent.llm_backend.provider,
                        description=agent.llm_backend.description,
                    )
                if agent.agents_max_tokens:
                    agent_configs[agent.name].agents_max_tokens = agent.agents_max_tokens

        # check MCP health on startup to mark if tools are loaded
        active_configs = dict(agent_configs)
        for agent_name, health_url in mcp_agents.items():
            if agent_name in active_configs:
                healthy = await wait_for_health(
                    url=health_url,
                    headers={"Accept": "application/json"},
                    success_codes=[200],
                    wait_seconds=10.0,
                    interval=3.0,
                    request_timeout=5.0,
                )
                if not healthy:
                    logger.warning("%s MCP unavailable at startup — agent will have no tools", agent_name)
                    ChatConsumer._tools_loaded[agent_name] = False
                else:
                    ChatConsumer._tools_loaded[agent_name] = True

        ChatConsumer.redbox = Redbox(
            agents=active_configs,
            env=ChatConsumer.env,
            debug=ChatConsumer.debug,
            sso_access_token=sso_access_token,
        )

    @staticmethod
    async def _mcp_monitor_loop(
        agent_name: str,
        mcp_url: str,
        get_token: Callable[[], str | None],
    ):
        """
        Background task - poll MCP health every 30s
        If MCP server state changes, reinitialise Redbox tools

        Inputs:
            agent_name: str - name of the MCP agent to check
            mcp_url: str - health endpoint URL of MCP server
            get_token: Callable - callable for generating a token
        """
        was_healthy: bool = True

        while True:
            await asyncio.sleep(30)
            try:
                healthy = await wait_for_health(
                    url=mcp_url,
                    headers={"Accept": "application/json"},
                    success_codes=[200],
                    wait_seconds=10,
                    interval=2,
                    request_timeout=3.0,
                )

                if healthy and not was_healthy:
                    # MCP came back up — reload tools
                    logger.info("%s MCP recovered — reloading tools", agent_name)

                    if not ChatConsumer._tools_loaded.get(agent_name):
                        # tools never loaded — full reinit required
                        logger.info("Tools were never loaded for %s — forcing full Redbox reinit", agent_name)
                        ChatConsumer.redbox = None  # force _init_redbox on next connect
                    elif ChatConsumer.redbox is not None:
                        try:
                            await ChatConsumer.redbox.reload_tools(agent=agent_name, sso_access_token=get_token())
                            ChatConsumer._tools_loaded[agent_name] = True
                        except AttributeError:
                            ChatConsumer.redbox = None
                    else:
                        # redbox was None despite tools previously being loaded — force reinit
                        logger.warning(
                            "%s redbox is None despite tools being marked loaded — forcing reinit", agent_name
                        )
                        ChatConsumer._tools_loaded[agent_name] = False

                elif not healthy and was_healthy:
                    # MCP went down — log it
                    logger.warning("%s MCP went down — tools unavailable until recovery", agent_name)

                was_healthy = healthy

            except Exception:
                logger.exception("%s MCP monitor error - ", agent_name)

    async def _reload_mcp_tools(self):
        """Reload redbox MCP tools"""
        if ChatConsumer.redbox is None:
            await self._init_redbox()
            return

        await ChatConsumer.redbox.reload_tools("Datahub_Agent", self._extract_sso_token())

    def update_chat_consumer_redbox_with_new_sso_token(self, sso_access_token: str) -> None:
        ChatConsumer.redbox.sso_access_token = sso_access_token
        ChatConsumer.redbox.init_datahub_agent(sso_access_token)
        ChatConsumer.redbox.setup_graph(ChatConsumer.debug)

    async def handle_text(self, response: str) -> str:
        """Handle text chunks and British spelling conversion before sending to client."""
        logger.debug("Received text chunk: %s", response)
        if getattr(self, "uk_english", False):
            converted_chunk = await sync_to_async(uwm8.convert_american_to_british_spelling, thread_sensitive=False)(
                response
            )
        else:
            converted_chunk = response

        # store both original and converted chunks
        self.full_reply.append(response)
        self.converted_reply.append(converted_chunk)

        # send converted text to client
        await self.send_to_client("text", converted_chunk)
        return converted_chunk

    async def handle_route(self, response: str) -> str:
        await self.send_to_client("route", response)
        self.route = response
        return response

    async def handle_metadata(self, response: dict):
        self.metadata = metadata_reducer(self.metadata, RequestMetadata.model_validate(response))

    async def handle_activity(self, response: dict):
        # Feature flag activity event display till design is confirmed
        if await sync_to_async(flag_is_active)(self.user, flags.ENABLE_ACTIVITY_EVENTS):
            await self.send_to_client("activity", response.message)
        self.activities.append(RedboxActivityEvent.model_validate(response))

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
                file = await self.get_file_cached(ref)
                if file:
                    payload = {"url": str(file.url), "file_name": file.file_name}
                else:
                    # If no file with Status.complete is found, handle it as None
                    payload = {"url": ref, "file_name": None}

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
                payload = {"url": ref, "file_name": None}
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

    async def handle_citations(self, citations: list[AICitation]):
        """
        Map AICitations used to create answer to AICitations for storing as citations. The link to user files
        must be populated
        """
        for c in citations:
            for s in c.sources:
                try:
                    # Use the async database query function
                    file = await self.get_file_cached(s.source)
                    if file:
                        payload = {
                            "url": str(file.url),
                            "file_name": file.file_name,
                            "citation_name": s.ref_id,
                        }
                    else:
                        # if source is empty, attempt to filter by document name
                        file = await self.get_file_cached(s.document_name)
                        if file:
                            payload = {
                                "url": str(file.url),
                                "file_name": file.file_name,
                                "citation_name": s.ref_id,
                            }
                        else:
                            # If no file with Status.complete is found, handle it as None
                            payload = {
                                "url": s.source,
                                "file_name": s.source,
                                "citation_name": s.ref_id,
                            }
                except File.DoesNotExist:
                    file = None
                    payload = {
                        "url": s.source,
                        "file_name": s.source,
                        "citation_name": s.ref_id,
                    }

                await self.send_to_client("source", payload)

                self.citations.append(
                    (
                        file,
                        AICitation(
                            sources=[s],
                        ),
                    )
                )
