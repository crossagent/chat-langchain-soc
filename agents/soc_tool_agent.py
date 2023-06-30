from langchain.agents import (
    AgentExecutor,
    LLMSingleActionAgent,
)
from langchain import OpenAI, LLMChain
from typing import List, Union
from outputParser.tooslsOutputParser import ToolsOutputParser
from prompts.prompt import (CustomPromptTemplate, template)
from retrievers.toolRetriever import get_tools


llm = OpenAI(temperature=0)

output_parser = ToolsOutputParser()

prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"],
)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tools = get_tools("whats the weather?")
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)