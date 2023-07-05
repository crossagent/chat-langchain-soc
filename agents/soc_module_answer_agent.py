from langchain.agents import (
    AgentExecutor,
    LLMSingleActionAgent,
)
from langchain import OpenAI, LLMChain
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from typing import List, Union
from outputParser.tooslsOutputParser import ToolsOutputParser
from prompts.prompt import (CustomPromptTemplate, template)
from retrievers.projectDocumentRetriever import get_retriver
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.base import BaseCallbackHandler
from typing import Optional, List, Union, Dict, Any
from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know because there is no relation data in our knowladgebase now, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

class RetrievalQACallBack(BaseCallbackHandler):

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Run when chain starts running."""

        print(f"除了咨询接口人信息，我会在知识库中进一步为您查询相关信息，帮助你解决问题。")

def get_expert_answer(llm, verbose = False) ->RetrievalQA:
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=get_retriver(), 
                                     chain_type_kwargs=chain_type_kwargs, verbose=verbose,
                                        callbacks=[RetrievalQACallBack()],)
    return qa