def create_retriever():
    from chromadb import PersistentClient as Client

    client = Client(path='./chroma_db')
    try:
        collection = client.get_collection(name='retriever_collection')
    except Exception:
        collection = client.create_collection(name='retriever_collection')
    return collection
