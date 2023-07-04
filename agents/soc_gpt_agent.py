from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.agents import initialize_agent
from chains.soc_question_generate import get_question_generate
from langchain.agents.agent import AgentExecutor
from pydantic import BaseModel, Field

memory = ConversationBufferMemory(memory_key="chat_history")

class QuestionInput(BaseModel):
    question: str = Field()
    
def get_agent(llm, verbose=False, callback_manager=None) -> AgentExecutor:
    question_agent = get_question_generate(llm, verbose=verbose, callback_manager=callback_manager)

    tools = [
    Tool(
        name=f"Problem solving helper",
        func=question_agent.run,
        description=f"help developer to solve problem during soc game development",
        args_schema=QuestionInput,
    )
    ]

    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=verbose, memory=memory)
    return agent_chain



def runTest():
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(temperature=0)

    agent = get_agent(llm, verbose=True)
    
    finalquestion = agent.run(input = "狙击枪坏掉了")
    print(finalquestion)

    finalquestion = agent.run(input = "另外人也没法跑步了")
    print(finalquestion)

if __name__ == "__main__":
    runTest()