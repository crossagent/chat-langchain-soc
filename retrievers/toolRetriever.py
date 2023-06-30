#from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings

from retrievers.pineconeclient import initPinecone
from tools.tools import ALL_TOOLS

docs = [
    Document(page_content=t.description, metadata={"index": i})
    for i, t in enumerate(ALL_TOOLS)
]


initPinecone()
vector_store = Pinecone.from_existing_index(index_name="rules", embedding=OpenAIEmbeddings(), text_key='text', namespace="experts")

retriever = vector_store.as_retriever()

def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    return [ALL_TOOLS[int(d.metadata["index"])] for d in docs]

tools = get_tools("whats the weather?")

print(tools)
