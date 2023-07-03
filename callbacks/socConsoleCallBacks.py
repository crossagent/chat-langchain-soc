"""Callback handlers used in the app."""
from typing import Any, Dict, List

from langchain.callbacks.base import AsyncCallbackHandler

from schemas import ChatResponse

class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token)


class QuestionGenCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Run when a chat model starts running."""
        print("question generation start")

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        print("start new chain")

        
    async def on_llm_start(
        self, serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        print("question generation start")