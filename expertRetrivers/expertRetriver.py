from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone


personal_texts = [
    "I love apple pie",
    "My favorite color is fuchsia",
    "My dream is to become a professional dancer",
    "I broke my arm when I was 12",
    "My parents are from Peru",
]
personal_retriever = Pinecone.from_texts(personal_texts, OpenAIEmbeddings()).as_retriever()

retriever_infos = [
    {
        "name": "state of the union", 
        "description": "Good for answering questions about the 2023 State of the Union address", 
        "retriever": personal_retriever
    },
    {
        "name": "pg essay", 
        "description": "Good for answer quesitons about Paul Graham's essay on his career", 
        "retriever": personal_retriever
    },
    {
        "name": "personal", 
        "description": "Good for answering questions about me", 
        "retriever": personal_retriever
    }
]