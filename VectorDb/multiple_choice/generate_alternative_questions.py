from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
import pandas as pd

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
import json_repair
import json
import gc

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

def log_memory_usage(step):
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"Step {step}: CUDA Memory Allocated: {memory_allocated:.2f} MB, Reserved: {memory_reserved:.2f} MB")

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
    return tokenizer, llm

def get_generation_chain(llm_model, prompt_template):
    SYSTEM_MESSAGE = """
    You are an AI assistant specialized in generating questions and multiple-choice answers based on provided context about university courses.
    Your task is to create engaging, thought-provoking questions and a set of answer choices that effectively test understanding of the material.
    Key Requirements:
    1. Generate one question and four answer choices (A, B, C, D) based on the given context.
    2. Ensure that exactly one answer choice is correct and directly supported by the context.
    3. Create three incorrect answers that are plausible but subtly misinterpret or alter details from the context.
    4. The incorrect answers should be challenging and require careful reading of the context to distinguish from the correct answer.
    5. If the context includes statistics, dates, or numerical data, incorporate these into the question and answers, subtly modifying values for incorrect options.
    6. Phrase the question and answers to test higher-order thinking skills (e.g., application, analysis, evaluation) rather than mere recall.
    8. Write all text in Italian.
    Output Format:
    Return your response in a structured JSON format with the following properties:
    - question: The generated question
    - correct_answer: The correct answer (labeled as A)
    - incorrect_answer_1: First incorrect answer (labeled as B)
    - incorrect_answer_2: Second incorrect answer (labeled as C)
    - incorrect_answer_3: Third incorrect answer (labeled as D)
    """

    USER_MESSAGE = """
    Generate a question and multiple-choice answers based on the following context about a university course.
    Adhere strictly to the requirements.
    <context>
    {context}
    </context>
    Expected JSON format:
    {{
        "question": "Your generated question about the given context",
        "correct_answer": "A. The correct answer for the question, extracted from the context",
        "incorrect_answer_1": "B. Your first misleading but plausible answer here.",
        "incorrect_answer_2": "C. Your second misleading but plausible answer here.",
        "incorrect_answer_3": "D. Your third misleading but plausible answer here."
    }}
    """
    # SYSTEM_MESSAGE = """
    # You are an AI assistant specialized in generating questions and multiple-choice answers based on provided context about university courses.
    # Your task is to create engaging, thought-provoking questions and sets of answer choices that effectively test understanding of the material.

    # Key Requirements:
    # 1. Generate as many questions as possible based on the given context, with each question having four answer choices (A, B, C, D).
    # 2. For each question, ensure that exactly one answer choice is correct and directly supported by the context.
    # 3. Create three incorrect answers for each question that are plausible but subtly misinterpret or alter details from the context.
    # 4. The incorrect answers should be challenging and require careful reading of the context to distinguish from the correct answer.
    # 5. If the context includes statistics, dates, or numerical data, incorporate these into questions and answers, subtly modifying values for incorrect options.
    # 6. Phrase the questions and answers to test higher-order thinking skills (e.g., application, analysis, evaluation) rather than mere recall.
    # 8. Write all text in Italian.

    # Output Format:
    # Return your response in a structured JSON format with an array of question objects, each containing the following properties:
    # - question: The generated question
    # - correct_answer: The correct answer (labeled as A)
    # - incorrect_answer_1: First incorrect answer (labeled as B)
    # - incorrect_answer_2: Second incorrect answer (labeled as C)
    # - incorrect_answer_3: Third incorrect answer (labeled as D)
    # """

    # USER_MESSAGE = """
    # Generate as many questions and multiple-choice answers as possible based on the following context about a university course.
    # Adhere strictly to the requirements.

    # <context>
    # {context}
    # </context>

    # Expected JSON format:
    # {{
    # "questions": [
    #     {{
    #     "question": "Your first generated question about the given context",
    #     "correct_answer": "A. The correct answer for the question, extracted from the context",
    #     "incorrect_answer_1": "B. Your first misleading but plausible answer here.",
    #     "incorrect_answer_2": "C. Your second misleading but plausible answer here.",
    #     "incorrect_answer_3": "D. Your third misleading but plausible answer here."
    #     }},
    #     {{
    #     "question": "Your second generated question about the given context",
    #     "correct_answer": "A. The correct answer for the question, extracted from the context",
    #     "incorrect_answer_1": "B. Your first misleading but plausible answer here.",
    #     "incorrect_answer_2": "C. Your second misleading but plausible answer here.",
    #     "incorrect_answer_3": "D. Your third misleading but plausible answer here."
    #     }},
    #     // ... additional questions ...
    #     ]
    # }}
    # """

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
        template=final_prompt_template, input_variables=["context"]
    )

    rag_chain = (
        {
            "context": RunnablePassthrough()
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

        tokenizer, llm = get_hugging_face_model(model_id=model_name, hf_token = hf_token)

        for collection_name, faq_link in collection_tuples:
            
            rag_chain = get_generation_chain(llm, model_prompt_template)

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

@torch.no_grad()
def get_valid_response(rag_chain, context, retry_limit=3):
    for _ in range(retry_limit):
        response = rag_chain.invoke({"context": context})
        gc.collect()
        torch.cuda.empty_cache()
        try:
            parsed_json = json_repair.loads(response)
            
            # Validate that all required keys are present
            if all(key in parsed_json for key in ["question", "correct_answer", "incorrect_answer_1", "incorrect_answer_2", "incorrect_answer_3"]):
                return parsed_json
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
    
    logger.warning(f"Failed to get valid response after {retry_limit} attempts.")
    return None

def chunk_text(text, tokenizer, max_tokens=1000, overlap=200):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk))
    return chunks

@torch.no_grad()
def debug():
    #Testing parameters
    HF_TOKEN = "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"
    FOLDER_NAME = "matematica"
    LLM_MODEL_NAME = "google/gemma-7b-it"
    GEMMA7B_PROMPT_TEMPLATE = ModelPromptTemplate(
        user_start="<start_of_turn>user\n",
        user_end="<end_of_turn>\n",
        assistant_start="<start_of_turn>model\n",
        assistant_end="<end_of_turn>\n",
        text_start="<bos>",
        text_end="")

    tokenizer, llm = get_hugging_face_model(model_id=LLM_MODEL_NAME, hf_token = HF_TOKEN) #Download model

    rag_chain = get_generation_chain(llm, GEMMA7B_PROMPT_TEMPLATE)

    documents = readDocuments("../" + FOLDER_NAME)
    logger.info(len(documents))
    # context = """
    # L'elefante è un mammifero di grandi dimensioni, noto per la sua maestosità e forza. È caratterizzato da un corpo massiccio coperto da una pelle spessa e grigiastra, grandi orecchie a forma di ventaglio e una lunga proboscide, che è un'estensione del naso e del labbro superiore. Questa proboscide è molto versatile e viene utilizzata per molte funzioni, come raccogliere cibo, bere acqua e comunicare.
    # Gli elefanti hanno anche grandi zanne d'avorio, che sono in realtà denti incisivi allungati e possono essere usate per scavare o difendersi. Le loro gambe sono forti e robuste, sostenendo il peso del loro enorme corpo, e i piedi hanno ampi cuscinetti che aiutano a distribuire il peso e a camminare silenziosamente nonostante la loro mole.
    # Esistono due specie principali di elefanti: l'elefante africano, che è generalmente più grande e ha orecchie più ampie, e l'elefante asiatico, che è leggermente più piccolo e ha orecchie più piccole. Gli elefanti vivono in gruppi sociali matriarcali, guidati dalla femmina più anziana, e sono animali molto intelligenti e sociali, noti per le loro forti connessioni familiari e per la loro capacità di provare emozioni come la gioia e il lutto."""

    OUTPUT_FOLDER_NAME = "questions"
    file_name = f"{FOLDER_NAME}.csv"
    complete_path = os.path.join(OUTPUT_FOLDER_NAME, FOLDER_NAME, file_name)
    os.makedirs(os.path.join(OUTPUT_FOLDER_NAME, FOLDER_NAME), exist_ok=True)
    output_df = pd.DataFrame(columns=['sorgente', 'domanda', 'corretta', 'errata_1', 'errata_2', 'errata_3'])
    for i, document in enumerate(documents):
        logger.info(document.metadata["file_name"])
        chunks = chunk_text(document.page_content, tokenizer)
        for j, chunk in enumerate(chunks):
            log_memory_usage("")
            parsed_json = get_valid_response(rag_chain, document.page_content, retry_limit=5)
            
            if parsed_json:
                output_df.loc[i] = [
                    document.metadata["file_name"],
                    parsed_json["question"],
                    parsed_json["correct_answer"],
                    parsed_json["incorrect_answer_1"],
                    parsed_json["incorrect_answer_2"],
                    parsed_json["incorrect_answer_3"]
                ]
            else:
                output_df.loc[i] = [
                    str(i) + "_" + document.metadata["file_name"],
                    "vuoto",
                    "vuoto",
                    "vuoto",
                    "vuoto",
                    "vuoto"
                ]
                logger.error(f"Failed to retrieve valid data for document {document.metadata['file_name']}")
    output_df.to_csv(complete_path, index=False)

if __name__ == "__main__":
    DEBUGGING = True
    with torch.no_grad():
        if not DEBUGGING:
            main()
        else:
            debug()