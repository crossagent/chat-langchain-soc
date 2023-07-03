#from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStoreRetriever
from retrievers.pineconeclient import initPinecone
from tools.tools import ALL_TOOLS

docs = [
    Document(page_content=t.description, metadata={"index": i})
    for i, t in enumerate(ALL_TOOLS)
]


initPinecone()
vector_store = Pinecone.from_existing_index(index_name="rules", embedding=OpenAIEmbeddings(), text_key='text', namespace="soc-project-0615")


def get_retriver() -> VectorStoreRetriever:
    retriever = vector_store.as_retriever()
    return retriever

def get_documents(query):
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(query)

    return docs

def runTest():
    docs = get_documents("如何查bug")

    for doc in docs:
        print(doc.page_content)
