from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.agents import initialize_agent
from tools.tools import ALL_TOOLS
from soc_question_agent import agent_executor

memory = ConversationBufferMemory(memory_key="chat_history")

tools = [
    Tool(
        name=f"Problem solving helper",
        func=agent_executor.run,
        description=f"help developer to solve problem during soc game development",
    )
]

llm=OpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

def get_agent():
    return agent_chain

agent_chain.run("我的狙击枪无法射击了")