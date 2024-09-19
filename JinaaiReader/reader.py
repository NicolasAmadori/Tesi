from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from milvus import default_server
from pymilvus import connections, utility, Collection

import glob
import os
from tqdm import tqdm
import json
import argparse

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from uuid import uuid4

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def create_db(data_path: str, 
              collection_name : str,
              embedder_model : HuggingFaceEmbeddings,
              text_splitter : RecursiveCharacterTextSplitter):
    try:
        utility.drop_collection(collection_name)
        logger.info("Dropped previous data")
    except:
        pass
    
    logger.info("Starting data ingestion")

    documents = []
    for file in tqdm(glob.glob(os.path.join(data_path, "*"))):
        if os.path.isdir(file): continue
        with open(file, 'r') as f:
            data = f.read()

            metadata = {
                "source" : "Unibo.it",
                "user" : "root",
                "file_name" : file
            }
            
            documents.append(Document(page_content=data, metadata=metadata))

    logger.info("Creating chunks")
    docs = text_splitter.split_documents(documents)

    logger.info("Embedding data and creating DB at http://milvus-standalone:19530")
    _ = Milvus.from_documents(
        docs,
        embedder_model,
        connection_args={"uri":  "http://milvus-standalone:19530"}, collection_name=collection_name
    )
    logger.info("Completed with success")

def create_milvus_db(embedder_model,
    collection_name = "UniboIngScInf",    
    data_path = "local/unibo_data",
    chunk_size = 128,
    chunk_overlap = 50) -> str:

    # >> Establish connection
    URI = "localhost:19530"
    logger.info("Starting the server at localhost:19530")
    default_server.start()
    connections.connect("default", uri=URI)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap,
                                                   separators=["\n\n"," ",".",","])

    # >> Start ingestion
    logger.info("Creating a new Milvus DB")
    create_db(data_path, collection_name=collection_name, embedder_model=embedder_model, text_splitter=text_splitter)

    collection = Collection(collection_name)
    logger.info(f'collection: {collection.name} has {collection.num_entities} entities')

    logger.info("Done")
    return URI

def main():
    embeddings = HuggingFaceEmbeddings(model="jinaai/jina-embeddings-v3", model_kwargs={"device": "cuda"})

    uri = create_milvus_db(embeddings, data_path = "")

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": uri},
    )

    vector_store_saved = Milvus.from_documents(
        [Document(page_content="foo!")],
        embeddings,
        collection_name="langchain_example",
        connection_args={"uri": uri},
    )

    vector_store_loaded = Milvus(
        embeddings,
        connection_args={"uri": uri},
        collection_name="langchain_example",
    )

    # documents = [
    #     document_1,
    #     document_2,
    #     document_3,
    #     document_4,
    #     document_5,
    #     document_6,
    #     document_7,
    #     document_8,
    #     document_9,
    #     document_10,
    # ]
    # uuids = [str(uuid4()) for _ in range(len(documents))]
    # vector_store.add_documents(documents=documents, ids=uuids)

if __name__ == "__main__":
    main()