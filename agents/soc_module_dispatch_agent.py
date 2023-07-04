from typing import Optional, List, Union, Dict, Any
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent import AgentExecutor
from langchain.chains.router import MultiRetrievalQAChain
from tools.modeleInfo import ALL_MODULE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI
from agents.soc_module_answer_agent import get_expert_answer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def startModuleSearch(*args, **kwargs) -> str:
    print("args:", args)
    print("kwargs:", kwargs)
    return "you can call zhongyu for help" # 返回模块名字和负责人

llm = OpenAI()

prompt = PromptTemplate(
    template="""没有合适的应答模块""",
    input_variables=[],
)


def get_desination_chains() -> Dict[str, RetrievalQA]:
    destination_chains = {}
    for p_info in ALL_MODULE:
        name = p_info["name"]
        chain = get_expert_answer(llm, verbose=True)
        destination_chains[name] = chain
    return destination_chains
        
default_chain = get_expert_answer(llm, verbose=True)

def get_module_dispatch_agent(llm, verbose=False, callback_manager=None) -> AgentExecutor:
    destinations = [f"{p['name']}: {p['description']}" for p in ALL_MODULE]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(next_inputs_inner_key="query"),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    chain = MultiRetrievalQAChain(
        router_chain=router_chain,
        destination_chains=get_desination_chains(),
        default_chain=default_chain,
        verbose=verbose,
    )

    return chain

def runTest():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    agent = get_module_dispatch_agent(llm, verbose=True)
    rest = agent.run("如何提bug")
    print(rest)

if __name__ == "__main__":
    runTest()