from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from uuid import uuid4

import logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    embedder_model_name = "BAAI/bge-m3" #Default: "jinaai/jina-embeddings-v3"
    embedder = HuggingFaceEmbeddings(model_name=embedder_model_name,
                                     model_kwargs={"device": "cuda"})

    vector_store = Milvus(
        embedding_function=embedder,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        collection_name="UniboIngScInf"
    )
        
    # # Ottieni l'elenco delle collezioni
    # collections = vector_store.list_collections()
    # logger.info(f"Collections presenti: {collections}")

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    response = retriever.invoke("Tirocinio", filter={"source": "Unibo.it"}, ef=30)
    logger.info(response)

if __name__ == "__main__":
    main()