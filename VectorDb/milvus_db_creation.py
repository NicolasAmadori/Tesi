from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker

from milvus import default_server
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

import glob
import os
from tqdm import tqdm
import json
import argparse

from transformers import AutoModel

import logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def create_db(data_path: str, 
              collection_name : str,
              embedder_model,
              text_splitter : RecursiveCharacterTextSplitter):
    try:
        utility.drop_collection(collection_name)
        logger.info("Dropped previous data")
    except:
        pass
    
    logger.info("Starting data ingestion")

    documents = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                data = f.read()

                metadata = {
                    "source": "Unibo.it",
                    "user": "root",
                    "file_name": file_path
                }
                documents.append(Document(page_content=data, metadata=metadata))

    logger.info("Creating chunks")
    docs = text_splitter.split_documents(documents)
    logger.info(f"Splitted documents from {len(documents)} to {len(docs)}")
    
    logger.info("Embedding data and creating DB at localhost:19530")
    _ = Milvus.from_documents(
        docs,
        embedder_model,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        collection_name=collection_name
    )
    logger.info("Completed with success")

def main():
    data_path = "IngegneriaScienzeInformatiche"
    collection_name = "UniboIngScInf"

    add_to_existing_collection = False #Default: False
    embedder_model_name = "BAAI/bge-m3" #Default: "jinaai/jina-embeddings-v3"
    chunk_size = 128 #Default: 256
    chunk_overlap = 50 #Default: 100

    # >> Establish connection
    try:
        connections.connect("default", host="127.0.0.1", port="19530")
    except:
        logger.info("Starting the server at 127.0.0.1")
        default_server.start()
        connections.connect("default", host="127.0.0.1", port="19530")

    # >> Load Model
    logger.info("Loading encoder model")
    embedder = HuggingFaceEmbeddings(model_name=embedder_model_name,
                                     model_kwargs={"device": "cuda"})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap,
                                                   separators=["\n\n"," ",".",","])

    # >> Start ingestion
    logger.info("Creating a new Milvus DB")
    create_db(data_path, collection_name=collection_name, embedder_model=embedder, text_splitter=text_splitter)

    logger.info("Checking if the collection was created")

    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        logger.info(f'Collection: {collection.name} has {collection.num_entities} entities')
    else:
        logger.error(f"Collection {collection_name} was not created")

    logger.info("Done")

if __name__ == "__main__":
    main()