"""Callback handlers used in the app."""
from typing import Any, Dict, List

from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler

from schemas import ChatResponse

class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token)

class StreamLLMCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="")    

class QuestionGenCallbackHandler(BaseCallbackHandler):
    """Callback handler for question generation."""

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Run when a chat model starts running."""
        print("正在收集对话信息，生成问题...")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        print(f"您的问题可以描述为：{outputs['text']}")