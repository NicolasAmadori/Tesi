from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker

from milvus import default_server
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
import torch

import os

import pandas as pd

import logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def chunker(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def readDocuments(folder_name, strings_to_remove = []):
    #Read files, remove faq questions and create documents
    collection_documents = []
    for root, _, files in os.walk(folder_name):#Iterate directories
        for file in files:#Iterate directory files
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                data = f.read()
                # for string in strings_to_remove:
                #     data = data.replace(string,"")
                
                metadata = {
                    "source": f"corsi.unibo.it/laurea/{folder_name}",
                    "file_name": file_path
                }
                
                collection_documents.append(Document(page_content=data, metadata=metadata))
    return collection_documents

@torch.no_grad()
def addCollectionDocumentsToDB(documents, text_splitter, embedding_model, host, port):
    DOCUMENT_CHUNK_SIZE = 256
    logger.info("Creating chunks and uploading documents to db")
    for collection_name, collection_documents in documents:
        splitted_documents = text_splitter.split_documents(collection_documents)
        logger.info(f"{collection_name} -> from {len(collection_documents)} to {len(splitted_documents)} documents")

        total = 0
        for chunk in chunker(splitted_documents, DOCUMENT_CHUNK_SIZE):
            total += DOCUMENT_CHUNK_SIZE
            logger.info(f"{min(total, len(splitted_documents))}")
            _ = Milvus.from_documents(
                chunk,
                embedding_model,
                connection_args={"host": host, "port": port},
                collection_name=collection_name
            )

def create_db(folders_collection_pairs,
              embedding_model,
              text_splitter : RecursiveCharacterTextSplitter,
              host = "0.0.0.0",
              port= "19530"):
    logger.info("Starting data ingestion")

    documents = [] #List of pair (collection_name, collection_documents)
    for folder_name, collection_name, collection_faqs_link in folders_collection_pairs:
        faqs_questions = pd.read_csv(collection_faqs_link)["domanda"].tolist() #Get the list of FAQ questions

        collection_documents = readDocuments(folder_name, faqs_questions) #Read documents and remove faqs questions from the data
        documents.append((collection_name, collection_documents))
        logger.info(f"{collection_name} -> {len(collection_documents)}")
    
    addCollectionDocumentsToDB(documents, text_splitter, embedding_model, host, port)
    logger.info("Completed with success")

def reset(collection_name):
    # Ensure the collection exists
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        collection.drop()
        print(f"Collection '{collection_name}' has been deleted.")
    else:
        print(f"Collection '{collection_name}' does not exist.")

def main():
    #(folder_name, collection_name, collection_faqs)
    collection_triplets = [
    #    ("IngegneriaScienzeInformatiche", "UniboIngScInf", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_ING_TRI.csv"),
        #  ("SviluppoCooperazioneInternazionale", "UniboSviCoop", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_COOP_TRI.csv"),
        ("matematica", "UniboMat", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_MAT_TRI.csv"),
        ]
    
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3" #Default: "jinaai/jina-embeddings-v3"
    CHUNK_SIZE = 512 #Default: 256
    CHUNK_OVERLAP = 200 #Default: 100
    HOST = "0.0.0.0"
    PORT = "19530"

    # >> Establish connection
    try:
        connections.connect("default", host=HOST, port=PORT)
    except:
        logger.info(f"Starting the server at {HOST}:{PORT}")
        default_server.start()
        connections.connect("default", host=HOST, port=PORT)

    # >> Load Model
    logger.info("Loading encoder model")
    embedding_model = None
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                     model_kwargs={"device": "cuda"})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, 
                                                   chunk_overlap=CHUNK_OVERLAP,
                                                   separators=["\n\n"," ",".",","])
    
    for _, collection_name, _ in collection_triplets:
        reset(collection_name)
    # >> Start ingestion
    logger.info("Uploading documents to Milvus DB")
    create_db(collection_triplets, embedding_model=embedding_model, text_splitter=text_splitter, host=HOST, port=PORT)
    torch.cuda.empty_cache()
    
    logger.info("Checking if collections were created")
    for _, collection_name, _ in collection_triplets:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            logger.info(f'Collection: {collection.name} has been created.')
        else:
            logger.error(f"Collection {collection_name} was not created")

    logger.info("Everything done")

if __name__ == "__main__":
    main()
