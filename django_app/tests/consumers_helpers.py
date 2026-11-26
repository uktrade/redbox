from channels.testing import WebsocketCommunicator
from django.utils import timezone
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from redbox_app.redbox_core.consumers import ChatConsumer


class Token(BaseModel):
    content: str


class ListContentToken(BaseModel):
    content: list


class CannedGraphLLM(BaseChatModel):
    responses: list[dict]

    def _generate(self, *_args, **_kwargs):
        for _ in self.responses:
            yield

    def _llm_type(self):
        return "canned"

    def _convert_input(self, prompt):
        if isinstance(prompt, dict):
            prompt = prompt["request"].question
        return super()._convert_input(prompt)

    async def astream_events(self, *_args, **_kwargs):
        for response in self.responses:
            yield response


async def connect_authenticated_chat(user):
    communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
    communicator.scope["user"] = user
    communicator.scope["session"] = {
        "_authbroker_token": {
            "access_token": "test-sso-token",
            "expires_at": timezone.now().timestamp() + 3600,
        }
    }

    connected, _ = await communicator.connect(timeout=5)
    assert connected

    ready = await communicator.receive_json_from(timeout=5)
    assert ready["type"] == "chat_ready"

    return communicator
