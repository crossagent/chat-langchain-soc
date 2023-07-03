import pinecone

initialized = False

def initPinecone() -> pinecone:
    pinecone.init(api_key="881ad1dd-9ab5-4406-baea-6a64f034c64e", environment="asia-southeast1-gcp-free")
    initialized = True
    return pinecone