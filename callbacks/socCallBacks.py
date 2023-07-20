"""Callback handlers used in the app."""
from typing import Any, Dict, List

from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler

from schemas import ChatResponse
import asyncio
import json

class StreamingSocLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket = None):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if (self.websocket):
            resp = ChatResponse(sender="bot", message=token, type="stream")
            await self.websocket.send_json(resp.dict())
        else:
            print(token)

class StreamSocLLMCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket = None):
        self.websocket = websocket

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if (self.websocket):        
            resp = ChatResponse(sender="bot", message=token, type="stream")
            asyncio.run(self.websocket.send_json(resp.dict()))
        else:
            print(token)    

class ChainSocCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    def __init__(self, websocket = None):
        self.websocket = websocket

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if (self.websocket):
            """Run when chain starts running."""  
            #输出AI的处理过程
            start_resp = ChatResponse(sender="bot", message="", type="start")
            asyncio.run(self.websocket.send_json(start_resp.dict()))

            stream_resp = ChatResponse(sender="bot", message="我开始思考，并制定计划。</br>", type="stream")
            asyncio.run(self.websocket.send_json(stream_resp.dict()))
        else:
            print("我开始思考，并制定计划")


    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """结束的时候把回答中的speak字段输出"""
        if (self.websocket):        
            # 获取 thoughts 下的 speak 字段的值
            json_data = json.loads(outputs['text'])
            
            thoughts = json_data["thoughts"]["text"]
            reasoning = json_data["thoughts"]["reasoning"]

            stream_resp = ChatResponse(sender="bot", message=f"我想：{thoughts} + </br>", type="stream")
            #asyncio.run(self.websocket.send_json(stream_resp.dict()))  

            stream_resp = ChatResponse(sender="bot", message=f"原因：{reasoning} + </br>", type="stream")
            #asyncio.run(self.websocket.send_json(stream_resp.dict()))  

            # 首先结束当前的对话
            end_resp = ChatResponse(sender="bot", message="", type="end")
            asyncio.run(self.websocket.send_json(end_resp.dict())) 

            # 开启一个新的对话说明结果
            start_resp = ChatResponse(sender="bot", message="", type="start")
            asyncio.run(self.websocket.send_json(start_resp.dict()))

            speak_field = json_data["thoughts"]["speak"]

            # 发送最终的询问结果
            result_resp = ChatResponse(sender="bot", message=speak_field, type="stream")
            asyncio.run(self.websocket.send_json(result_resp.dict()))

            # 结束对话
            asyncio.run(self.websocket.send_json(end_resp.dict())) 
        else:
            print(outputs['text'])        
        

class QuestionGenSocCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket = None):
        self.websocket = websocket

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        resp = ChatResponse(
            sender="bot", message="Synthesizing question...", type="info"
        )
        await self.websocket.send_json(resp.dict())

class QuestionRephSocCallbackHandler(BaseCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket = None):
        self.websocket = websocket

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        # 首先结束当前的对话
        end_resp = ChatResponse(sender="bot", message="", type="end")
        asyncio.run(self.websocket.send_json(end_resp.dict())) 

        # 开启一个新的对话说明结果
        start_resp = ChatResponse(sender="bot", message="", type="start")
        asyncio.run(self.websocket.send_json(start_resp.dict()))

        # 发送最终的询问结果
        #result_resp = ChatResponse(sender="bot", message=speak_field, type="stream")
        #asyncio.run(self.websocket.send_json(result_resp.dict()))

        # 结束对话
        asyncio.run(self.websocket.send_json(end_resp.dict()))     

class ToolUseCallbackkHandler(BaseCallbackHandler):

    def __init__(self, websocket = None):
        self.websocket = websocket

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        if self.websocket:
            """Run when tool starts running."""
            # 首先结束当前的对话
            end_resp = ChatResponse(sender="bot", message="", type="end")
            asyncio.run(self.websocket.send_json(end_resp.dict())) 

            # 开启一个新的对话说明问题
            start_resp = ChatResponse(sender="bot", message="", type="start")
            asyncio.run(self.websocket.send_json(start_resp.dict()))

            # 发送最终的询问结果
            eval_input = eval(input_str)

            result_resp = ChatResponse(sender="bot", message=f"从网络搜索提问：{eval_input['question']}", type="stream")
            asyncio.run(self.websocket.send_json(result_resp.dict()))

            asyncio.run(self.websocket.send_json(end_resp.dict())) 

            # 开启一个新的对话说明结果
            start_resp = ChatResponse(sender="bot", message="", type="start")
            asyncio.run(self.websocket.send_json(start_resp.dict()))
        else:
            print(input_str)

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        if self.websocket:
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
        else:
            print(output)

from langchain.schema import Document

class WebSearchCallbackHandler(BaseCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket = None):
        self.websocket = websocket

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""
        if self.websocket:
            input_documents = inputs['input_documents']

            if len(input_documents) > 0:
                metadata = input_documents[0].metadata
                source = metadata['source']

                # 发送最终的询问结果
                result_resp = ChatResponse(sender="bot", message=f"</br>我正在浏览{source}", type="stream")
                asyncio.run(self.websocket.send_json(result_resp.dict()))
        else:
            print(inputs['input_documents'])

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        if self.websocket:
            output_text = outputs['output_text']

            # 发送最终的询问结果
            result_resp = ChatResponse(sender="bot", message=f"</br>根据这部分内容：{output_text}", type="stream")
            asyncio.run(self.websocket.send_json(result_resp.dict()))

            # 结束对话
            end_resp = ChatResponse(sender="bot", message="", type="end")
            asyncio.run(self.websocket.send_json(end_resp.dict())) 
        else:
            print(outputs['output_text'])            

