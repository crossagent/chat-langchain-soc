"""Load html from files, clean up, split, ingest into Weaviate."""
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from retrievers.pineconeclient import initPinecone
from langchain.vectorstores.pinecone import Pinecone
from langchain.schema import Document
from tools.tools import ALL_TOOLS
import pinecone

docs = [
    Document(page_content=t.description, metadata={"index": i})
    for i, t in enumerate(ALL_TOOLS)
]


def ingest_expert():
    """Get documents from web pages."""
    pinecone = initPinecone()

    index = pinecone.Index("rules")
    delete_response = index.delete(deleteAll='true', namespace="experts")

    vector_store = Pinecone.from_documents(docs, OpenAIEmbeddings(), index_name="rules", namespace="experts")

if __name__ == "__main__":
    ingest_expert()
