from typing import Optional, List, Union, Dict, Any
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
from langchain.memory import ConversationBufferWindowMemory

# 问题模板
# 提问中关于玩家当前的描述，要包含{职业}{正在执行的任务}{遇到的问题详细描述}{需要的帮助类型}
# 首先你需要明确问题，output回答中需要包含json格式的内容
# 根据分析，tools中包含所有的问题模块，你的问题属于XX模块，正在查询相关模块的信息

def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)

human = HumanInputRun()

tools = [
    Tool(
        name="HumanInput",
        func=human.run,
        description="useful for when you need to find the perpose of the question"
    )
]

# Set up the base template
template_with_history = """You are an AI assistant, helping people to solve problem during game development work. \

The questioner is a {profession}, You need to clarify the user's problem through tools, the task the user is performing, and the type of help the user needs.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input in chinese: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak in chinese when giving your Action Input and final answer.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class QuestionPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    # The profession of the user
    profession : str


    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # Add the profession to the kwargs
        kwargs["profession"] = self.profession
        tmplstr = self.template.format(**kwargs)
        return tmplstr

class QuestionOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*in\s*chinese\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
output_parser = QuestionOutputParser()

# LLM chain consisting of the LLM and a prompt

def get_question_agent(llm, 
                       callback_manager: Optional[BaseCallbackManager] = None, 
                       verbose=False, 
                       **kwargs,
                       ) -> AgentExecutor:
    prompt_with_history = QuestionPromptTemplate(
        template=template_with_history,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # profession add the sameway, in format function,don't use input_variables
        profession=kwargs["profession"],
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps", "history"]
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    memory=ConversationBufferWindowMemory(k=5)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, 
                                                        verbose=verbose, memory=memory, 
                                                        callback_manager=callback_manager)

    return agent_executor

async def runTest():
    from callbacks.socConsoleCallBacks import QuestionGenCallbackHandler
    from langchain.callbacks.manager import AsyncCallbackManager
    
    llm = ChatOpenAI()

    config = {"profession" : "game developer"}

    questionhandler = QuestionGenCallbackHandler()
    questionMgr = AsyncCallbackManager([questionhandler])
    agent = get_question_agent(llm, verbose=True, callback_manager=questionMgr, **config)
    
    finalquestion = await agent.arun("狙击枪坏掉了")

    print(finalquestion)

    from soc_module_expert_agent import get_expert_answer

    expert = get_expert_answer(llm)
    
    answer = await expert.arun(finalquestion)

    print(answer)

import asyncio
result = asyncio.run(runTest())