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



from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Chinese:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

def get_expert_answer(llm, verbose = False) ->RetrievalQA:
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=get_retriver(), chain_type_kwargs=chain_type_kwargs, verbose=verbose)
    return qa