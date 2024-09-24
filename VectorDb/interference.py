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
        connection_args={"host": "0.0.0.0", "port": "19530"},
        collection_name="UniboIngScInf"
    )
    
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "param": {"ef":30}})

    while True:
        try:
            question = input("Input: ")
        
            response = retriever.invoke(question, filter={"source": "Unibo.it"})
            logger.info(response)
        except Exception as e:
            logger.info(e)

if __name__ == "__main__":
    main()