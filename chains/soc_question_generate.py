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

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import BaseChatMessageHistory, HumanMessage, AIMessage
import json

class QuestionGenerateChain(LLMChain):

    full_memory : BaseChatMessageHistory = None

    memory : ConversationBufferMemory = None

    @classmethod
    def from_llm(cls, llm, verbose=False, callbacks=None):

        memory = ConversationBufferMemory(memory_key="chat_history")

        condense_question_prompt="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
        The standalone question is preferably expected to encompass the user's purpose, the context of the question, the type of assistance they require, and any additional requirements.

        for example:
        Chat History:
        Humam:i want to travel to hainan, what should i do?
        AI:give me more information about your travel plan.
        Humam:no i wan't low cost, i have only 10000 yuan.
        AI:give me more information about your travel plan.
        Humam:i will travel with my family, we have 3 people, and we have 7 days.
        Follow Up Input:i wan't a plan for travel?
        Standalone question:I am planning to travel to Hainan, with a budget of 10,000 yuan. There will be three people traveling, and we have a total of 7 days. I would like assistance on how to plan the itinerary and request a travel guide. Please note that I prefer a relaxed itinerary and avoid popular tourist attractions.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question in chinese:"""

        condense_question_prompt = PromptTemplate(
            template=condense_question_prompt,
            input_variables=["chat_history", "question"],
        )        

        return cls(llm=llm, prompt= condense_question_prompt, verbose=verbose, callbacks=callbacks, memory=memory)
    
    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:

        # 重新提取聊天记录
        self.memory.clear()
        # 提取外部的数据，作为聊天记忆
        if self.full_memory is not None:
            for message in self.full_memory.messages:
                if message.type == HumanMessage(content="").type:
                    content = message.content

                    if (False == content.startswith("Determine which next")):
                        self.memory.chat_memory.add_message(HumanMessage(content=content))

                elif message.type == AIMessage(content="").type:
                    if 'thoughts' in message.content:
                        content = json.loads(message.content)
                        #thoughts = content['thoughts']['text']
                        #reasoning = content['thoughts']['reasoning']
                        #new_content = str({'thoughts': thoughts, 'reasoning': reasoning})
                        #self.memory.chat_memory.add_message(AIMessage(content=new_content))

                        speak = content['thoughts']['speak']
                        self.memory.chat_memory.add_message(AIMessage(content=speak))
            
            messages_str = str(self.memory.chat_memory.messages)

            print("将从这些聊天记忆提取问题\n" + messages_str)

        return super().prep_inputs(inputs)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        return super()._call(inputs, run_manager=run_manager)

def runTest():
    from callbacks.socConsoleCallBacks import QuestionGenCallbackHandler
    from langchain.callbacks.manager import BaseCallbackManager
    
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    questionhandler = QuestionGenCallbackHandler()
    questionMgr = BaseCallbackManager([questionhandler])
    
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