from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.human.tool import HumanInputRun
from typing import Any, Callable, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from uuid import UUID
from fastapi import WebSocket
import asyncio
from schemas import ChatResponse

async def WaitUserInput(websocket: WebSocket):
    #首先中断AI的输入
    #end_resp = ChatResponse(sender="bot", message="", type="end")
    #await websocket.send_json(end_resp.dict())

    #开始等待玩家的输入
    question = await websocket.receive_text()
    resp = ChatResponse(sender="you", message=question, type="stream")
    await websocket.send_json(resp.dict())

    return question

def get_user_input(websocket: WebSocket) -> str:
    return asyncio.run(WaitUserInput(websocket))


class WebHumanInputRun(HumanInputRun):
    
    websocket : WebSocket = None

    def __init__(self, websocket):
        super().__init__()
        self.websocket = websocket
        self.input_func = get_user_input
        self.callbacks = [CallForHumanCallbackHandler(websocket)]

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Human input tool."""
        self.prompt_func(query)
        return self.input_func(self.websocket)

class CallForHumanCallbackHandler(BaseCallbackHandler):

    def __init__(self, websocket):
        self.websocket = websocket 

    """Callback handler for question generation."""
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        """Run when a chat model starts running."""
