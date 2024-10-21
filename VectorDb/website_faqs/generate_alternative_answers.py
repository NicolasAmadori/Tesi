from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
import pandas as pd

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
import json_repair

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
        # quantization_config=BitsAndBytesConfig(load_in_4bit=True),
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

def get_rag_chain(llm_model, prompt_template):
    SYSTEM_MESSAGE = """You are an AI assistant tasked with generating incorrect answers to questions based on the given context. 
    Your goal is to create two **plausible yet false answers** that subtly misinterpret the context or alter details in a way that seems credible but is ultimately incorrect when compared to the accurate information. 

    IMPORTANT: 
    - The false answers should be **misleading** but not absurd, making them **reasonable enough to confuse the reader**.
    - Derive these incorrect answers **only from the provided context** by subtly changing facts, figures, or interpretations.
    - If statistics or numerical data appear in the context, use them but **modify values** in a realistic way to craft the false answers.
    - The answers must be strictly written in **Italian**.

    Return your output in a structured JSON format as specified below.

    """

    USER_MESSAGE = """Based on the provided context, generate two false but plausible answers to the following question. 
    IMPORTANT: 
    - Both answers must be **incorrect** but seem **believable** at first glance.
    - Only use the information found in the context to **slightly alter** or misinterpret it.
    - If the context includes numbers or statistics, try to incorporate these by modifying or interpreting them differently.
    - STRICTLY return the output in the format of a JSON object, with two properties: `false_answer_1` and `false_answer_2`.
    - Answers should be written strictly in **Italian**.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    <correct_answer>
    {correct_answer}
    </correct_answer>

    Expected JSON format:

    {{
        "false_answer_1": "Your first misleading but plausible answer here.",
        "false_answer_2": "Your second misleading but plausible answer here."
    }}
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

        llm = get_hugging_face_model(model_id=model_name, hf_token = hf_token)

        for collection_name, faq_link in collection_tuples:
            
            rag_chain = get_rag_chain(llm, model_prompt_template)

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
        "google/gemma-2-9b-it":GEMMA7B_PROMPT_TEMPLATE
    }

    generate_answers(EMBEDDING_MODEL_NAME, COLLECTION_TUPLES, TESTING_MODEL_DICT, HF_TOKEN, k=8)

def debug():
    #Testing parameters
    HF_TOKEN = "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"
    FOLDER_NAME = "IngegneriaScienzeInformatiche"
    FAQ_LINK = "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_ING_TRI.csv"
    LLM_MODEL_NAME = "google/gemma-2-9b-it"
    GEMMA7B_PROMPT_TEMPLATE = ModelPromptTemplate(
        user_start="<start_of_turn>user\n",
        user_end="<end_of_turn>\n",
        assistant_start="<start_of_turn>model\n",
        assistant_end="<end_of_turn>\n",
        text_start="<bos>",
        text_end="")

    llm = get_hugging_face_model(model_id=LLM_MODEL_NAME, hf_token = HF_TOKEN) #Download model

    rag_chain = get_rag_chain(llm, GEMMA7B_PROMPT_TEMPLATE)

    documents = readDocuments(FOLDER_NAME)
    faq_dataframe = pd.read_csv(FAQ_LINK)

    for index, (domanda, risposta, origine) in enumerate(zip(faq_dataframe["domanda"], faq_dataframe["risposta"], faq_dataframe["origine"]), 1):
        if index > 100000:
            break
        logger.info(f"{index} -> {origine}")
        # d = getDocument(documents, origine)

        # response = rag_chain.invoke({
        #     "context": d.page_content,
        #     "question": domanda,
        #     "correct_answer": risposta
        # })
        # logger.info(response)

    #Domanda test
    context = """
    L'elefante è un mammifero di grandi dimensioni, noto per la sua maestosità e forza. È caratterizzato da un corpo massiccio coperto da una pelle spessa e grigiastra, grandi orecchie a forma di ventaglio e una lunga proboscide, che è un'estensione del naso e del labbro superiore. Questa proboscide è molto versatile e viene utilizzata per molte funzioni, come raccogliere cibo, bere acqua e comunicare.
    Gli elefanti hanno anche grandi zanne d'avorio, che sono in realtà denti incisivi allungati e possono essere usate per scavare o difendersi. Le loro gambe sono forti e robuste, sostenendo il peso del loro enorme corpo, e i piedi hanno ampi cuscinetti che aiutano a distribuire il peso e a camminare silenziosamente nonostante la loro mole.
    Esistono due specie principali di elefanti: l'elefante africano, che è generalmente più grande e ha orecchie più ampie, e l'elefante asiatico, che è leggermente più piccolo e ha orecchie più piccole. Gli elefanti vivono in gruppi sociali matriarcali, guidati dalla femmina più anziana, e sono animali molto intelligenti e sociali, noti per le loro forti connessioni familiari e per la loro capacità di provare emozioni come la gioia e il lutto."""

    domanda = "Quante specie principali di elefanti esistono?"
    risposta = "Esistono due specie principali di elefanti."
    d = Document(page_content=context)

    #Prima FAQ
    faq_index = 32
    domanda, risposta, origine = faq_dataframe["domanda"].tolist()[faq_index], faq_dataframe["risposta"].tolist()[faq_index], faq_dataframe["origine"].tolist()[faq_index]
    d = getDocument(documents, origine)
    logger.info(domanda)
    logger.info(risposta)
    logger.info(d.metadata)

    while True:
        input("Premi invio per generare")
        response = rag_chain.invoke({
            "context": d.page_content,
            "question": domanda,
            "correct_answer": risposta
        })
        
        parsed_json = json_repair.loads(response)
        logger.info(f"Original -> {response}")
        logger.info(f"Parsed JSON -> {parsed_json}")
    
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