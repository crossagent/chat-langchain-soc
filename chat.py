from langchain.chat_models import ChatOpenAI
from agents.soc_gpt_agent import get_agent
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.agents import AgentExecutor
from typing import Any, Dict, List
from agents.soc_module_answer_agent import get_expert_answer
from fastapi import WebSocket
from agents.front_desk_agent import get_front_dest_agent, FrontDestAgent

def get_web_front_desk_agent(websocket: WebSocket) -> FrontDestAgent:
    from callbacks.socCallBacks import StreamSocLLMCallbackHandler, ChainSocCallbackHandler, ToolUseCallbackkHandler
    from tools.WebHumanInputRun import WebHumanInputRun

    stream_handler = StreamSocLLMCallbackHandler(websocket)
    input_run = WebHumanInputRun(websocket)
    tool_handler = ToolUseCallbackkHandler(websocket)
    agent_hander = ChainSocCallbackHandler(websocket)

    #soc_agent from chat.py
    config = {"stream_handler" : stream_handler, "feedback_tool": input_run, "rust_tool_handler":tool_handler, "agent_hander":agent_hander}
    soc_agent = get_front_dest_agent(**config)

    return soc_agent


def get_console_front_desk_agent() -> FrontDestAgent:
    from callbacks.socCallBacks import StreamSocLLMCallbackHandler, ChainSocCallbackHandler, ToolUseCallbackkHandler
    from langchain.tools.human.tool import HumanInputRun

    stream_handler = StreamSocLLMCallbackHandler()
    input_run = HumanInputRun()
    tool_handler = ToolUseCallbackkHandler()
    agent_hander = ChainSocCallbackHandler()

    #soc_agent from chat.py
    config = {"stream_handler" : stream_handler, "feedback_tool": input_run, "rust_tool_handler":tool_handler, "agent_hander":agent_hander}
    soc_agent = get_front_dest_agent(**config)

    return soc_agent

if __name__ == "__main__":
    soc_agent = get_console_front_desk_agent()

    import asyncio
    print("输入你的问题：")
    input = input()
    asyncio.run(soc_agent.arun([input], count=5))