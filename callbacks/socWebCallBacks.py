"""Callback handlers used in the app."""
from typing import Any, Dict, List

from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler

from schemas import ChatResponse
import asyncio
import json

class StreamingSocLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message=token, type="stream")
        await self.websocket.send_json(resp.dict())


class StreamSocLLMCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message=token, type="stream")
        asyncio.run(self.websocket.send_json(resp.dict()))

class ChainSocCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    def __init__(self, websocket):
        self.websocket = websocket

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""  
        #输出AI的处理过程
        start_resp = ChatResponse(sender="bot", message="", type="start")
        asyncio.run(self.websocket.send_json(start_resp.dict()))

        stream_resp = ChatResponse(sender="bot", message="我开始思考，并制定计划。\n", type="stream")
        asyncio.run(self.websocket.send_json(stream_resp.dict()))


    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """结束的时候把回答中的speak字段输出"""
        # 首先结束当前的对话
        end_resp = ChatResponse(sender="bot", message="", type="end")
        asyncio.run(self.websocket.send_json(end_resp.dict())) 

        # 开启一个新的对话说明结果
        start_resp = ChatResponse(sender="bot", message="", type="start")
        asyncio.run(self.websocket.send_json(start_resp.dict()))

        # 获取 thoughts 下的 speak 字段的值
        json_data = json.loads(outputs['text'])
        speak_field = json_data["thoughts"]["speak"]

        # 发送最终的询问结果
        result_resp = ChatResponse(sender="bot", message=speak_field, type="stream")
        asyncio.run(self.websocket.send_json(result_resp.dict()))

        # 结束对话
        asyncio.run(self.websocket.send_json(end_resp.dict()))         
        

class QuestionGenSocCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        resp = ChatResponse(
            sender="bot", message="Synthesizing question...", type="info"
        )
        await self.websocket.send_json(resp.dict())


class ToolUseCallbackkHandler(BaseCallbackHandler):

    def __init__(self, websocket):
        self.websocket = websocket

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        # 首先结束当前的对话
        end_resp = ChatResponse(sender="bot", message="", type="end")
        asyncio.run(self.websocket.send_json(end_resp.dict())) 

        # 开启一个新的对话说明结果
        start_resp = ChatResponse(sender="bot", message="", type="start")
        asyncio.run(self.websocket.send_json(start_resp.dict()))

        # 发送最终的询问结果
        result_resp = ChatResponse(sender="bot", message="正在查询网站，并阅读网页内容\n", type="stream")
        asyncio.run(self.websocket.send_json(result_resp.dict()))

                # 结束对话
        end_resp = ChatResponse(sender="bot", message="", type="end")
        asyncio.run(self.websocket.send_json(end_resp.dict())) 

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""

        # 首先结束当前的对话
        end_resp = ChatResponse(sender="bot", message="", type="end")
        asyncio.run(self.websocket.send_json(end_resp.dict())) 

        data = eval(output)

        # 开启一个新的对话说明结果
        start_resp = ChatResponse(sender="bot", message="", type="start")
        asyncio.run(self.websocket.send_json(start_resp.dict()))

        # 发送最终的询问结果
        result_resp = ChatResponse(sender="bot", message=data['output_text'], type="stream")
        asyncio.run(self.websocket.send_json(result_resp.dict()))

        # 结束对话
        asyncio.run(self.websocket.send_json(end_resp.dict())) 
