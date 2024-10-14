from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
import pandas as pd

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

#RAG Chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

#Logging
import logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

class Delimiter:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class ModelPromptTemplate:
    def __init__(self, system_start="", system_end="", user_start="", user_end="", assistant_start="", assistant_end="", text_start="", text_end=""):
        self.user_delimiter = Delimiter(user_start, user_end) if user_start != "" or user_end != "" else None
        self.assistant_delimiter = Delimiter(assistant_start, assistant_end) if assistant_start != "" or assistant_end != "" else None
        self.system_delimiter = Delimiter(system_start, system_end) if system_start != "" or system_end != "" else None
        self.text_delimiter = Delimiter(text_start, text_end) if text_start != "" or text_end != "" else None

def readDocuments(folder_name, strings_to_remove = []):
    #Read files, remove faq questions and create documents
    collection_documents = []
    for root, _, files in os.walk(folder_name):#Iterate directories
        for file in files:#Iterate directory files
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                data = f.read()
                for string in strings_to_remove:
                    data = data.replace(string,"")
                
                metadata = {
                    "source": f"corsi.unibo.it/laurea/{folder_name}",
                    "file_name": file_path
                }
                
                collection_documents.append(Document(page_content=data, metadata=metadata))
    return collection_documents

@torch.no_grad()
def get_hugging_face_model(model_id, hf_token, return_full_text = False):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="cuda",
        trust_remote_code = True, #Added for Phi-3-mini-128k
        token = hf_token,
        torch_dtype=torch.float16, #Aggiunto, da verificare la correttezza
        # attn_implementation="flash_attention_2" # if you have an ampere GPU (RTX3090 OK, T4(Colab) NON OK)
    )
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=1024,
                    top_k=50,
                    temperature=0.2,
                    return_full_text = return_full_text,
                    do_sample = True)
    llm = HuggingFacePipeline(pipeline=pipe,
                            pipeline_kwargs={"return_full_text": return_full_text}) # <----- IMPORTANT !!!
    return llm

def get_db_retriever(embedding_model, collection_name, host="0.0.0.0", port="19530", search_type="mmr", k=4):
    vector_store = Milvus(
        embedding_function=embedding_model,
        connection_args={"host": host, "port": port},
        collection_name=collection_name
    )

    return vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})

def get_rag_chain(llm_model, prompt_template, retriever):
    SYSTEM_MESSAGE = """You are an AI assistant, and your role is to generate incorrect answers to questions.
    When asked, generate two plausible but false answers based on the given context.
    The false answers should be sensible enough to seem correct at first glance but must be incorrect when compared with the accurate information.
    Use the context provided to generate the two answers, ensuring that them are misleading but not absurd.
    IMPORTANT: you have to generate the answers strictly in italian."""

    USER_MESSAGE = """Use the given context to generate an answer for the question.
    IMPORTANT: Generate two false but plausible answers that are incorrect but reasonable enough to confuse the reader, strictly in Italian
    IMPORTANT: Use only the information you can find in the given context to craft answers, and derive the two incorrect answers by slightly altering or misinterpreting the context in subtle ways.
    IMPORTANT: If statistics or numbers are present in the context and are useful for the answer, try to include them in the answers (with different values or interpretations).

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    <correct_answer>
    {correct_answer}
    </correct_answer>
    """

    final_prompt_template = ""
    if prompt_template.text_delimiter:
        final_prompt_template += prompt_template.text_delimiter.start

    if prompt_template.system_delimiter:
        final_prompt_template+= f"{prompt_template.system_delimiter.start}{SYSTEM_MESSAGE}{prompt_template.system_delimiter.end}"
        final_prompt_template+= f"{prompt_template.user_delimiter.start}{USER_MESSAGE}{prompt_template.user_delimiter.end}"
    else:
        final_prompt_template+= f"{prompt_template.user_delimiter.start}{SYSTEM_MESSAGE}\n\n{USER_MESSAGE}{prompt_template.user_delimiter.end}"
    
    final_prompt_template+= f"{prompt_template.assistant_delimiter.start}"
    
    prompt = PromptTemplate(
        template=final_prompt_template, input_variables=["context", "question", "correct_answer"]
    )

    rag_chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
            "correct_answer": RunnablePassthrough()
        }
        | prompt
        | llm_model
        | StrOutputParser()
    )
    return rag_chain

def generate_collection_false_answers(collection_name, faq_dataframe, rag_chain, output_path):
    output_df = pd.DataFrame(columns=['domanda', 'risposta_gold', 'risposta'])
    n_rows = faq_dataframe.shape[0]
    logger.info(f"Generating answers for the collection: {collection_name} ({n_rows} rows)")

    for index, (domanda, risposta_gold) in enumerate(zip(faq_dataframe["domanda"], faq_dataframe["risposta"]), 1):
        logger.info(f"{index}/{n_rows}")
        risposte_generate = rag_chain.invoke(domanda, risposta_gold)
        # output_df.loc[index] = [domanda, risposta_gold, risposta_generata]
    
    #Create the output csv file
    file_name = f"{collection_name}.csv"
    complete_path = os.path.join(output_path, file_name)
    os.makedirs(output_path, exist_ok=True)
    output_df.to_csv(complete_path, index=False)
   
def generate_answers(embedding_model_name, collection_tuples, models_dict, hf_token, k=4):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
        model_kwargs={"device": "cuda"})

    for model_name, model_prompt_template in models_dict.items():
        logger.info(f"Starting answers generation using: {model_name}")

        llm = get_hugging_face_model(model_id=model_name, hf_token = hf_token) #Download model

        #Generate answers for all the collections
        for collection_name, faq_link in collection_tuples:
            
            #Create rag_chain
            retriever = get_db_retriever(embedding_model, collection_name, k=k)
            rag_chain = get_rag_chain(llm, model_prompt_template, retriever)

            faq_dataframe = pd.read_csv(faq_link)
            generate_collection_false_answers(collection_name, faq_dataframe, rag_chain, output_path=f"output/{model_name}")
        
        torch.cuda.empty_cache()

def getDocument(documents, page_url):
    URL_PREFIX = 'https://corsi.unibo.it/laurea/'
    if page_url.startswith(URL_PREFIX):
        page_url = page_url.replace(URL_PREFIX, '')

    PDF_SUFFIX = ".pdf"
    if page_url.endswith(PDF_SUFFIX):
        page_url = page_url.replace(PDF_SUFFIX, PDF_SUFFIX + '.txt')

    extension = os.path.splitext(page_url)[1]
    if not extension:
        page_url += "/index.txt"

    for d in documents:
        source = d.metadata["file_name"]
        if source == page_url:
            return d
    
    raise Exception("No document with the given url found")

def main():
    HF_TOKEN = "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"
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3" #Default: "jinaai/jina-embeddings-v3"

    #(collection_name, collection_faqs)
    COLLECTION_TUPLES = [
        ("UniboIngScInf", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_ING_TRI.csv"),
        # ("UniboSviCoop", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_COOP_TRI.csv"),
        # ("UniboMat", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_MAT_TRI.csv")
        ]
    
    GEMMA7B_PROMPT_TEMPLATE = ModelPromptTemplate(
        user_start="<start_of_turn>user\n",
        user_end="<end_of_turn>",
        assistant_start="<start_of_turn>model",
        assistant_end="<end_of_turn>",
        text_start="<bos>",
        text_end="")
    
    TESTING_MODEL_DICT = {
        "google/gemma-7b-it":GEMMA7B_PROMPT_TEMPLATE
    }

    generate_answers(EMBEDDING_MODEL_NAME, COLLECTION_TUPLES, TESTING_MODEL_DICT, HF_TOKEN, k=8)

def debug():
    #Testing parameters
    HF_TOKEN = "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"
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3" #Default: "jinaai/jina-embeddings-v3"
    COLLECTION_NAME = "UniboIngScInf"
    FOLDER_NAME = "IngegneriaScienzeInformatiche"
    FAQ_LINK = "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_ING_TRI.csv"
    LLM_MODEL_NAME = "google/gemma-7b-it"
    GEMMA7B_PROMPT_TEMPLATE = ModelPromptTemplate(
        user_start="<start_of_turn>user\n",
        user_end="<end_of_turn>",
        assistant_start="<start_of_turn>model",
        assistant_end="<end_of_turn>",
        text_start="<bos>",
        text_end="")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"})

    llm = get_hugging_face_model(model_id=LLM_MODEL_NAME, hf_token = HF_TOKEN) #Download model

    retriever = get_db_retriever(embedding_model, COLLECTION_NAME, k=8)
    rag_chain = get_rag_chain(llm, GEMMA7B_PROMPT_TEMPLATE, retriever)

    documents = readDocuments(FOLDER_NAME)
    faq_dataframe = pd.read_csv(FAQ_LINK)

    for index, (domanda, risposta, origine) in enumerate(zip(faq_dataframe["domanda"], faq_dataframe["risposta"], faq_dataframe["origine"]), 1):
        if index > 1:
            break

        d = getDocument(documents, origine)

        response = rag_chain.invoke(context=d.page_content, question=domanda, correct_answer=risposta)
        logger.info(response) 

    # try:
    #     response = rag_chain.invoke(domanda, risposta)
    #     logger.info(response)         
    # except Exception as e:
    #     logger.error(f"Error in debug loop: {str(e)}", exc_info=True)
    #     print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    DEBUGGING = True
    if not DEBUGGING:
        main()
    else:
        debug()