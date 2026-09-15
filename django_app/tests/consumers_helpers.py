from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel


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
