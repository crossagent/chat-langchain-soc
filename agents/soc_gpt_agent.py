from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.agents import initialize_agent
from tools.tools import ALL_TOOLS
from agents.soc_question_agent import get_question_agent

memory = ConversationBufferMemory(memory_key="chat_history")


def get_agent(llm, verbose=False, callback_manager=None):
    question_agent = get_question_agent(llm, verbose=verbose, callback_manager=callback_manager)

    tools = [
    Tool(
        name=f"Problem solving helper",
        func=question_agent.run,
        description=f"help developer to solve problem during soc game development",
    )
    ]

    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=verbose, memory=memory)
    return agent_chain