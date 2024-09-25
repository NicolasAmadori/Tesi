from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from uuid import uuid4
import pandas as pd

import logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def stampa_righe_con_null(df, name):
    print(name)
    for index, row in df.iterrows():
        if row.isnull().any():  # Controlla se c'è almeno un valore null nella riga
            print(f"Riga {index} contiene un valore null:")
            print(row)
            print()

def main():
    faq_ing_url = "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_ING_TRI.csv"
    faq_coop_url = "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_COOP_TRI.csv"
    faq_mat_url = "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_MAT_TRI.csv"
    
    faq_ing = pd.read_csv(faq_ing_url)
    faq_coop = pd.read_csv(faq_coop_url)
    faq_mat = pd.read_csv(faq_mat_url)

    # Controllo valori nulli nei tre DataFrame
    stampa_righe_con_null(faq_ing, "FAQ ING")
    stampa_righe_con_null(faq_coop, "FAQ COOP")
    stampa_righe_con_null(faq_mat, "FAQ MAT")

    # embedder_model_name = "BAAI/bge-m3" #Default: "jinaai/jina-embeddings-v3"
    # embedder = HuggingFaceEmbeddings(model_name=embedder_model_name,
    #                                  model_kwargs={"device": "cuda"})

    # vector_store = Milvus(
    #     embedding_function=embedder,
    #     connection_args={"host": "0.0.0.0", "port": "19530"},
    #     collection_name="UniboIngScInf"
    # )
    
    # retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "param": {"ef":30}})

    # while True:
    #     try:
    #         question = input("Input: ")
        
    #         response = retriever.invoke(question, filter={"source": "Unibo.it"})
    #         logger.info(response)
    #     except Exception as e:
    #         logger.info(e)

if __name__ == "__main__":
    main()