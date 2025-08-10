import os

import chromadb
import typer
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from utils.ollama import OllamaEmbedding


def is_running_in_docker():
    return os.path.exists("/.dockerenv")

OLLAMA_URL = "http://localhost:11434" if not is_running_in_docker() else "http://ollama:11434"
CHROMA_URL = "localhost" if not is_running_in_docker() else "chroma"
CHROMA_PORT = 8000

ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_URL,
    ollama_additional_kwargs={"mirostat": 0},
)

chroma_client = chromadb.HttpClient(host=CHROMA_URL, port=CHROMA_PORT)

def search(
    query: str, 
    lib_name: str, 
    version: str
):   
    print(f"Searching for query '{query}' in {lib_name}:{version}")

    collection_name = f"{lib_name}_{version}"
    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=ollama_embedding
    )

    retriever_engine = index.as_retriever()
    response = retriever_engine.retrieve(query)

    result = response[0]
    print(result.score)
    print(result)

if __name__ == "__main__":
    typer.run(search)
            