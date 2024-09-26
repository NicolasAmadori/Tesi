from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
import pandas as pd

#RAG Chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Logging
import logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def getDBRetriever(embedding_model, collection_name, host="0.0.0.0", port="19530", search_type="mmr", k=4, ef=30):

    vector_store = Milvus(
        embedding_function=embedding_model,
        connection_args={"host": host, "port": port},
        collection_name=collection_name
    )

    return vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k, "param": {"ef":ef}})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def getRAGChain(llm_model, retriever):
    # Define the prompt template for generating AI responses
    PROMPT_TEMPLATE = """
    Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    The response should be specific and use statistics or numbers when possible.

    Assistant:"""

    # Create a PromptTemplate instance with the defined template and input variables
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    # Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )
    return rag_chain

def generate_answers(collection_name, faq_dataframe, rag_chain, output_path):
    output_df = pd.DataFrame(columns=['domanda', 'risposta_gold', 'risposta'])
    n_rows = dataframe.shape[0]
    logger.info(f"Generating answers for the collection: {collection_name} ({n_rows} rows)")
    for index, row in faq_dataframe.iterrows(): #Itera ogni riga del csv
        logger.info(f"{index+1}/{n_rows}")
        domanda = row["domanda"]
        risposta_gold = row["risposta"]
        risposta_generata = rag_chain.invoke(question)

        output_df.loc[index] = [domanda, risposta_gold, risposta_generata]
    
    #Create the output csv file
    file_name = f"{collection_name}.csv"
    output_df.to_csv(os.path.join(output_path, file_name), index=False)
   
def generate(embedding_model_name, collection_tuples, model_names_list):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
        model_kwargs={"device": "cuda"})

    for model_name in model_names_list:
        logger.info(f"Starting answers generation using: {model_name}")

        model = getHuggingFaceModel(model_id="meta-llama/Meta-Llama-3-8B-Instruct", hf_token = "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")#Download model

        #Generate answers for all the collections
        for collection_name, faq_link in collection_tuples:
            
            #Create rag_chain
            retriever = getDBRetriever(embedding_model, collection_name)
            rag_chain = getRAGChain(model, retriever)

            faq_dataframe = pd.read_csv(faq_link)
            generate_answers(collection_name, faq_dataframe, rag_chain, f"output/{model_name}")

def main():
    # EMBEDDING_MODEL_NAME = "BAAI/bge-m3" #Default: "jinaai/jina-embeddings-v3"

    # #(collection_name, collection_faqs)
    # COLLECTION_TUPLES = [
    #     ("UniboIngScInf", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_ING_TRI.csv"),
    #     ("UniboSviCoop", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_COOP_TRI.csv"),
    #     ("UniboMat", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_MAT_TRI.csv")]

    # TESTING_MODEL_NAMES = ["microsoft/Phi-3.5-mini-instruct",
    #     "meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     "mistralai/Mistral-7B-v0.3"]

    # generate(EMBEDDING_MODEL_NAME, COLLECTION_TUPLES, TESTING_MODEL_NAMES)
    
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
        
            response = retriever.invoke(question)
            logger.info(response)
        except Exception as e:
            logger.info(e)

if __name__ == "__main__":
    main()