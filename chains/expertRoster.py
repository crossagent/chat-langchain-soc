from typing import Any, Dict, List, Optional
from langchain.chains.router import MultiRetrievalQAChain
from langchain import PromptTemplate
from langchain.llms import BaseLLM
from langchain.chains.base import Chain
from langchain.schema import BaseRetriever

class SocExpertChain():
    """Chain to generate the next utterance for the conversation."""
    
    qaChain : MultiRetrievalQAChain

    @classmethod
    def from_retrievers(
        cls,
        llm: BaseLLM,
        retriever_infos: List[Dict[str, Any]],
        default_retriever: Optional[BaseRetriever] = None,
        default_prompt: Optional[PromptTemplate] = None,
        default_chain: Optional[Chain] = None,
        **kwargs: Any,
    ) -> "SocExpertChain":
        """Initialize the ConversationChain."""

        return MultiRetrievalQAChain.from_retrievers(
            llm=llm,
            retriever_infos=retriever_infos,
            default_retriever=default_retriever,
            default_prompt=default_prompt,
            default_chain=default_chain,
            **kwargs
        )
        