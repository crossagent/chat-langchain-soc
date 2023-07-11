from langchain.chat_models import ChatOpenAI
from agents.soc_gpt_agent import get_agent
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.agents import AgentExecutor
from typing import Any, Dict, List
from agents.soc_module_answer_agent import get_expert_answer


def get_soc_chain(
    question_handler:AsyncCallbackHandler = None, stream_handler:AsyncCallbackHandler = None, tracing: bool = False
) -> AgentExecutor:
    """Get the chain."""
    # init llm
    steam_manager = AsyncCallbackManager([stream_handler])
    llm = ChatOpenAI(temperature=0, streaming=True, callback_manager=steam_manager)

    soc_agent = get_agent(llm, verbose=False, callback_manager=[question_handler])

    return soc_agent

def get_soc_gpt(
    tracing: bool = False, 
    verbose: bool = False,
    **kwargs: Any
) -> AgentExecutor:
    """Get the chain."""
    # init llm
    steam_manager = AsyncCallbackManager([kwargs['stream_handler']])
    llm = ChatOpenAI(temperature=0, streaming=True, callback_manager=steam_manager)

    question_manager = AsyncCallbackManager(kwargs['question_handler'])
    soc_agent = get_agent(llm, verbose=verbose, callback_manager=[question_manager])

    return soc_agent

async def runTest():
    from callbacks.socConsoleCallBacks import QuestionGenCallbackHandler,StreamingLLMCallbackHandler

    question_handler = QuestionGenCallbackHandler()
    stream_handler = StreamingLLMCallbackHandler()

    config = {'stream_handler' : stream_handler, 'question_handler' : question_handler}
    agent_executor = get_soc_gpt(tracing=False, verbose=True, **config)
    
    result = await agent_executor.arun("狙击枪无法使用了")

import asyncio
result = asyncio.run(runTest())