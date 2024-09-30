from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
import pandas as pd

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from typing import List, Tuple, Dict

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

class Delimiter:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class ModelPromptTemplate:
    def __init__(self, system_start, system_end, user_start, user_end, assistant_start, assistant_end, text_start="", text_end=""):
        self.user_delimiter = Delimiter(user_start, user_end)
        self.assistant_delimiter = Delimiter(assistant_start, assistant_end)
        self.system_delimiter = Delimiter(system_start, system_end)
        self.text_delimiter = Delimiter(text_start, text_end) if text_start != "" or text_end != "" else None

def get_hugging_face_model(model_id, hf_token, return_full_text = False):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="cuda",
        trust_remote_code = True, #Added for Phi-3-mini-128k
        token = hf_token,
        torch_dtype=torch.float16 #Aggiunto, da verificare la correttezza
        #attn_implementation="flash_attention_2", # if you have an ampere GPU (RTX3090 OK, T4(Colab) NON OK)
    )
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=1024,
                    top_k=50,
                    temperature=0.1,
                    return_full_text = return_full_text)
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(llm_model, prompt_template, retriever):
    SYSTEM_MESSAGE = """You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer."""

    USER_MESSAGE = """Use the given context to generate an answer for the given question.
    IMPORTANT: Generate the answer strictly in italian.
    IMPORTANT: Use only the informations you can find in the given context.
    IMPORTANT: If statistics or numbers are present in the context and are useful for the answer, try to include them

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>
    """

    final_prompt_template = ""
    if prompt_template.text_delimiter:
        final_prompt_template += prompt_template.text_delimiter.start

    if prompt_template.system_delimiter.start and prompt_template.system_delimiter.start:
        final_prompt_template+= f"{prompt_template.system_delimiter.start}{SYSTEM_MESSAGE}{prompt_template.system_delimiter.end}"
        final_prompt_template+= f"{prompt_template.user_delimiter.start}{USER_MESSAGE}{prompt_template.user_delimiter.end}"
    else:
        final_prompt_template+= f"{prompt_template.user_delimiter.start}{SYSTEM_MESSAGE}\n\n{USER_MESSAGE}{prompt_template.user_delimiter.end}"
    
    final_prompt_template+= f"{prompt_template.assistant_delimiter.start}"

    if prompt_template.text_delimiter:
        final_prompt_template += prompt_template.text_delimiter.end
    
    prompt = PromptTemplate(
        template=final_prompt_template, input_variables=["context", "question"]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )
    return rag_chain

def generate_collection_answers(collection_name, faq_dataframe, rag_chain, output_path):
    output_df = pd.DataFrame(columns=['domanda', 'risposta_gold', 'risposta'])
    n_rows = faq_dataframe.shape[0]
    logger.info(f"Generating answers for the collection: {collection_name} ({n_rows} rows)")

    for index, row in faq_dataframe.iterrows(): #Itera ogni riga del csv
        logger.info(f"{index+1}/{n_rows}")
        domanda = row["domanda"]
        risposta_gold = row["risposta"]
        risposta_generata = rag_chain.invoke(domanda)

        output_df.loc[index] = [domanda, risposta_gold, risposta_generata]
    
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
            generate_collection_answers(collection_name, faq_dataframe, rag_chain, output_path=f"output/{model_name}")
        
        torch.cuda.empty_cache()

def main():
    HF_TOKEN = "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"
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3" #Default: "jinaai/jina-embeddings-v3"

    #(collection_name, collection_faqs)
    COLLECTION_TUPLES = [
        # ("UniboIngScInf", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_ING_TRI.csv"),
        ("UniboSviCoop", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_COOP_TRI.csv"),
        # ("UniboMat", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/FAQ/FAQ_MAT_TRI.csv")
        ]

    phi3_5_prompt_template = ModelPromptTemplate(
        system_start="<|system|>\n",
        system_end="<|end|>\n",
        user_start="<|user|>\n",
        user_end="<|end|>\n",
        assistant_start="<|assistant|>\n",
        assistant_end="<|end|>\n")

    llama3_1_prompt_template = ModelPromptTemplate(
        system_start="<|start_header_id|>system<|end_header_id|>\n\n",
        system_end="<|eot_id|>",
        user_start="<|start_header_id|>user<|end_header_id|>\n\n",
        user_end="<|eot_id|>",
        assistant_start="<|start_header_id|>assistant<|end_header_id|>\n\n",
        assistant_end="<|eot_id|>",
        text_start="<|begin_of_text|>")

    mistral0_3_prompt_template = ModelPromptTemplate(
        system_start="",
        system_end="",
        user_start="",
        user_end="",
        assistant_start="",
        assistant_end="",
        text_start="<s>[INST]",
        text_end="[/INST]")
    
    TESTING_MODEL_DICT = {
        # "microsoft/Phi-3.5-mini-instruct":phi3_5_prompt_template,
        # "meta-llama/Meta-Llama-3.1-8B-Instruct":llama3_1_prompt_template,
        "mistralai/Mistral-7B-Instruct-v0.3":mistral0_3_prompt_template
    }

    generate_answers(EMBEDDING_MODEL_NAME, COLLECTION_TUPLES, TESTING_MODEL_DICT, HF_TOKEN, k=8)

def debug():
    #Testing parameters
    HF_TOKEN = "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"
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3" #Default: "jinaai/jina-embeddings-v3"
    COLLECTION_NAME = "UniboIngScInf"
    LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    mistral0_3_prompt_template = ModelPromptTemplate(
        system_start="",
        system_end="",
        user_start="",
        user_end="",
        assistant_start="",
        assistant_end="",
        text_start="<s>[INST]",
        text_end="[/INST]")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"})

    llm = get_hugging_face_model(model_id=LLM_MODEL_NAME, hf_token = HF_TOKEN) #Download model

    retriever = get_db_retriever(embedding_model, COLLECTION_NAME, k=8)
    rag_chain = get_rag_chain(llm, mistral0_3_prompt_template, retriever)

    while True:
        try:
            question = input("Input: ")
            
            response = rag_chain.invoke(question)
            logger.info(response)

            # response = retriever.invoke(question)
            # total = ""
            # for doc in response:
            #     total+= doc.page_content + "\n\n"
            #     logger.info(doc.metadata["file_name"])
            # logger.info(total)            
        except Exception as e:
            logger.error(f"Error in debug loop: {str(e)}", exc_info=True)
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    DEBUGGING = False
    if not DEBUGGING:
        main()
    else:
        debug()