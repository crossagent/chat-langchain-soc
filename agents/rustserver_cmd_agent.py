# General
from langchain.chat_models import ChatOpenAI

from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
import asyncio

from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import BaseCallbackManager

from langchain.tools import Tool

# Memory
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.human.tool import HumanInputRun

embeddings_model = OpenAIEmbeddings()
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


from typing import List, Optional

from pydantic import ValidationError

from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from langchain.experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from langchain.experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.memory import ChatMessageHistory
from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    Document,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains.llm import LLMChain
from chains.soc_question_generate import QuestionGenerateChain

class RustServerGPT:
    """Agent class for interacting with Auto-GPT."""

    def __init__(
        self,
        ai_name: str,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        feedback_tool: Optional[HumanInputRun] = None,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool
        self.chat_history_memory = chat_history_memory or ChatMessageHistory()

    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        human_in_the_loop: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
        callbacks: Optional[List[BaseCallbackManager]] = None,
    ) -> "RustServerGPT":
        prompt = AutoGPTPrompt(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            input_variables=["memory", "messages", "goals", "user_input"],
            token_counter=llm.get_num_tokens,
        )
        human_feedback_tool = HumanInputRun() if human_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=prompt, callbacks=callbacks)
        return cls(
            ai_name,
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
            feedback_tool=human_feedback_tool,
            chat_history_memory=chat_history_memory,
        )

    async def arun(self, goals: List[str], count:int = 5) -> str:
        user_input = (
            "Determine which next command to use, "
            "and respond using the format specified above:"
        )

        self.chat_history_memory.add_message(HumanMessage(content="".join(goals)))

        # Interaction Loop
        loop_count = 0
        while loop_count < count:
            # Discontinue if continuous limit is reached
            loop_count += 1

            # Send message to AI, get response
            assistant_reply = self.chain.run(
                goals=goals,
                messages=self.chat_history_memory.messages,
                memory=self.memory,
                user_input=user_input,
            )

            # Print Assistant thoughts
            print(assistant_reply)
            self.chat_history_memory.add_message(HumanMessage(content=user_input))
            self.chat_history_memory.add_message(AIMessage(content=assistant_reply))

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                #return action.args["response"]
                #结束的时候不直接返回，等待用户决策
                pass
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                result = f"Command {tool.name} returned: {observation}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )


            humanNewInput = False

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            )
            if self.feedback_tool is not None:
                feedback = f"\n{self.feedback_tool.run('waiting for feedback')}"
                feedback = feedback.strip()
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    return "EXITING"
                elif feedback in {"c", "continue"}:
                    pass
                else:
                    memory_to_add += feedback
                    humanNewInput = True
                
            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.chat_history_memory.add_message(AIMessage(content=result))

            if humanNewInput:
                question_rephrase_chain = QuestionGenerateChain.from_llm(self.chain.llm, verbose = True)
                question_rephrase_chain.full_memory = self.chat_history_memory

                final_question = question_rephrase_chain.run(question = feedback)
                
                self.chat_history_memory.add_message(HumanMessage(content=final_question))


from tools.WebHumanInputRun import WebHumanInputRun, CallForHumanCallbackHandler
from tools.ServerCmdSearchTool import SeverGmCmdTool,load_qa_with_sources_chain
from callbacks.socCallBacks import ToolUseCallbackkHandler, WebSearchCallbackHandler, ChainSocCallbackHandler

def get_rust_server_cmd_gpt(
    verbose: bool = False,
    **kwargs: Any
) -> RustServerGPT:
    """Get the chain."""
    # init llm
    steam_manager = BaseCallbackManager([kwargs['stream_handler']])

    llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True, callback_manager = steam_manager)

    websocket = kwargs['websocket']

    llm_lookup = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", verbose= True)
    qa_manager = WebSearchCallbackHandler(websocket)

    from prompts.search_prompt import PROMPT as SEARCH_PROMPT
    qa_config = {"callbacks" : [qa_manager], "prompt" : SEARCH_PROMPT}
    qa_chain= load_qa_with_sources_chain(llm_lookup, **qa_config)

    llm_summary = ChatOpenAI(temperature=0, model="gpt-4")
    from prompts.summary_prompt import PROMPT as SUMMARY_PROMPT
    summary_config = {"prompt" : SUMMARY_PROMPT}
    summary_chain = load_qa_with_sources_chain(llm_summary, **summary_config)

    query_rust_tool = SeverGmCmdTool(qa_chain = qa_chain, summary_chain = summary_chain, callbacks = [ToolUseCallbackkHandler(websocket)], verbose = True)

    tools = [
        query_rust_tool,
        #human_input_tool, # Activate if you want the permit asking for help from the human
    ]    

    agent = RustServerGPT.from_llm_and_tools(
        ai_name="Tom",
        ai_role="Assistant",
        tools=tools,
        llm=llm,
        memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
        human_in_the_loop=True, # Set to True if you want to add feedback at each step.
        callbacks = [ChainSocCallbackHandler(websocket)],
    )
    agent.feedback_tool = WebHumanInputRun(websocket)
    agent.chain.verbose = True

    return agent

   
if __name__ == "__main__":
    from callbacks.socCallBacks import StreamLLMCallbackHandler

    stream_handler = StreamLLMCallbackHandler()

    #soc_agent from chat.py
    config = {"stream_handler" : stream_handler}
    soc_agent = get_rust_server_cmd_gpt(**config)

    asyncio.run(soc_agent.arun(["如何查看自己的位置"], count=5))

    
