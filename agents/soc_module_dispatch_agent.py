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
from agents.soc_module_answer_agent import get_expert_answer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

llm = ChatOpenAI()

default_prompt = PromptTemplate(
    template="""礼貌告知玩家没有找不到回答这个问题的模块""",
    input_variables=[],
)

class ModuleDispatchCallBack(BaseCallbackHandler):

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        if "destination" in outputs:
            module_name = outputs["destination"]
            module = ALL_MODULE[module_name]
            module_contact = module["contact_person"]
            progammer = module["progammer"]
            quality_assurance = module["quality_assurance"]

            # Continue executing the rest of the code

            print(f"该模块属于{module_name},主要负责的策划接口人是{module_contact},技术接口人是{progammer}, 质量保证是{quality_assurance}")


def get_desination_chains() -> Dict[str, RetrievalQA]:
    destination_chains = {}
    for p_info in ALL_MODULE.values():
        name = p_info["name"]
        chain = get_expert_answer(llm, verbose=False)
        destination_chains[name] = chain
    return destination_chains

def get_module_dispatch_agent(llm, verbose=False, callback_manager=None) -> AgentExecutor:
    destinations = [f"{p['name']}: {p['description']}" for p in ALL_MODULE.values()]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(next_inputs_inner_key="query"),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt, verbose=verbose, callbacks=[ModuleDispatchCallBack()])

    chain = MultiRetrievalQAChain(
        router_chain=router_chain,
        destination_chains=get_desination_chains(),
        default_chain=get_expert_answer(llm=llm, verbose=True),
        verbose=verbose,
    )

    return chain

def runTest():
    llm = ChatOpenAI(temperature=0)
    agent = get_module_dispatch_agent(llm, verbose=False)
    rest = agent.run("how to build a house")
    print(rest)

if __name__ == "__main__":
    runTest()