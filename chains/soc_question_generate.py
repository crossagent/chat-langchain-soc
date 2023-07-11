from typing import Optional, List, Union, Dict, Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain.agents import load_tools, initialize_agent
from langchain.tools import HumanInputRun
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# LLM chain consisting of the LLM and a prompt


class QuestionGenerateChain(LLMChain):

    memory : ConversationBufferMemory = None

    @classmethod
    def from_llm(cls, llm, verbose=False, callbacks=None):

        memory = ConversationBufferMemory(memory_key="chat_history")

        condense_question_prompt="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question in chinese:"""

        condense_question_prompt = PromptTemplate(
            template=condense_question_prompt,
            input_variables=["chat_history", "question"],
        )        

        return cls(llm=llm, prompt= condense_question_prompt, verbose=verbose, callbacks=callbacks, memory=memory)


def get_question_generate(llm, 
                       callbacks: Optional[Union[List[BaseCallbackHandler], BaseCallbackManager]] = None, 
                       verbose=False, 
                       **kwargs,
                       ) -> LLMChain:
    memory = ConversationBufferMemory(memory_key="chat_history")

    condense_question_prompt="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question in chinese:"""

    condense_question_prompt = PromptTemplate(
        template=condense_question_prompt,
        input_variables=["chat_history", "question"],
    )

    cpmdense_question_chain = LLMChain(
        llm=llm,
        prompt=condense_question_prompt,
        verbose=verbose,
        callbacks=callbacks,
        memory=memory,
    )

    return cpmdense_question_chain

def runTest():
    from callbacks.socConsoleCallBacks import QuestionGenCallbackHandler
    from langchain.callbacks.manager import BaseCallbackManager
    
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    questionhandler = QuestionGenCallbackHandler()
    questionMgr = BaseCallbackManager([questionhandler])
    #agent = get_question_generate(llm, verbose=True, callback_manager=questionMgr)
    
    agent = QuestionGenerateChain.from_llm(llm, verbose=False, callbacks=questionMgr)

    print("请输入提问：")
    question = input()
    finalquestion = agent.run(question)

    print("正在根据问题寻找合适的专家模块...")

    from agents.soc_module_dispatch_agent import get_module_dispatch_agent

    expert = get_module_dispatch_agent(llm, verbose=False)
    
    answer = expert.run(finalquestion)

    print(answer)

if __name__ == "__main__":
    runTest()