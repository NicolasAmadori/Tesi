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

#Logging
import logging
import random
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
        self.user_delimiter = Delimiter(user_start, user_end)
        self.assistant_delimiter = Delimiter(assistant_start, assistant_end)
        self.system_delimiter = Delimiter(system_start, system_end)
        self.text_delimiter = Delimiter(text_start, text_end)

def randomize_answers(answers):
    if len(answers) != 4:
        raise ValueError("The input must contain exactly 4 answers.")
    
    letters = ['A', 'B', 'C', 'D']
    random.shuffle(answers)  # Shuffle the letters randomly
    
    correct = None
    randomized_answers = []
    for new_letter, original_answer in zip(letters, answers):
        new_answer = f"{new_letter}. {original_answer[3:]}"
        if original_answer.startswith("A. "):
            correct = new_answer
        randomized_answers.append(new_answer)

    return randomized_answers, correct

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

def get_chain(llm_model, prompt_template):
    SYSTEM_MESSAGE = """You are an AI assistant and provide answers to questions.
    Return the correct answer to the question enclosed in <question> tags.
    For multiple-choice questions, respond strictly with one of the given options: A, B, C, or D.
    IMPORTANT: Do not provide any explanations or additional text."""

    USER_MESSAGE = """Generate an answer for the given multiple-choice question.
    IMPORTANT: Generate the answer strictly in Italian.
    IMPORTANT: Respond with only one of the 4 given options, including the letter and the text of the correct answer.
    IMPORTANT: Do not include any explanations or additional text in your response.

    <question>
    {question}
    </question>
    """

    final_prompt_template = prompt_template.text_delimiter.start

    if prompt_template.system_delimiter.start != "" or prompt_template.system_delimiter.end != "":
        final_prompt_template+= f"{prompt_template.system_delimiter.start}{SYSTEM_MESSAGE}{prompt_template.system_delimiter.end}"
        final_prompt_template+= f"{prompt_template.user_delimiter.start}{USER_MESSAGE}{prompt_template.user_delimiter.end}"
    else:
        final_prompt_template+= f"{prompt_template.user_delimiter.start}{SYSTEM_MESSAGE}\n\n{USER_MESSAGE}{prompt_template.user_delimiter.end}"
    
    final_prompt_template+= f"{prompt_template.assistant_delimiter.start}"
    final_prompt_template+= "The letter of the correct answer is: "
    
    prompt = PromptTemplate(
        template=final_prompt_template, input_variables=["question"]
    )

    chain = (
        {
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_model
        | StrOutputParser()
    )
    return chain

def generate_collection_answers(collection_name, faq_dataframe, chain, output_path):
    output_df = pd.DataFrame(columns=['domanda', 'risposta_gold', 'risposta', 'se_corretta'])
    n_rows = faq_dataframe.shape[0]
    logger.info(f"Generating answers for the collection: {collection_name} ({n_rows} rows)")

    file_name = f"{collection_name}.csv"
    complete_path = os.path.join(output_path, file_name)
    os.makedirs(output_path, exist_ok=True)

    for index, (domanda, corretta, errata_1, errata_2, errata_3) in enumerate(zip(faq_dataframe["domanda"], faq_dataframe["corretta"], faq_dataframe["errata_1"], faq_dataframe["errata_2"], faq_dataframe["errata_3"]), 1):
        logger.info(f"{index}/{n_rows}")

        randomized_answers, new_correct = randomize_answers([corretta, errata_1, errata_2, errata_3])
        query = domanda + "\n\n" + '\n'.join(randomized_answers)
        risposta_generata = chain.invoke(query)

        new_correct = new_correct.lstrip()
        risposta_generata = risposta_generata.lstrip()
        output_df.loc[index] = [domanda, new_correct, risposta_generata, new_correct.lower()[0] == risposta_generata.lower()[0]]
        logger.info(f"{domanda} -> {new_correct} -> {risposta_generata}")
        if index % 10 == 0:
            logger.info(f"Salvato il csv")
            output_df.to_csv(complete_path, index=False)#Intermidiate savings
    
    #Save the output csv file
    output_df.to_csv(complete_path, index=False)
   
def generate_answers(collection_tuples, models_dict, hf_token, k=4):

    for model_name, model_prompt_template in models_dict.items():
        logger.info(f"Starting answers generation using: {model_name}")

        llm = get_hugging_face_model(model_id=model_name, hf_token = hf_token) #Download model

        #Generate answers for all the collections
        for collection_name, faq_link in collection_tuples:
            
            #Create chain
            chain = get_chain(llm, model_prompt_template)

            faq_dataframe = pd.read_csv(faq_link)
            generate_collection_answers(collection_name, faq_dataframe, chain, output_path=f"answers_without_rag/{model_name}")
        
        torch.cuda.empty_cache()

def main():
    HF_TOKEN = "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"

    #(collection_name, collection_faqs)
    COLLECTION_TUPLES = [
        ("UniboIngScInf", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/VectorDb/multiple_choice/questions/IngegneriaScienzeInformatiche/IngegneriaScienzeInformatiche.csv"),
        ("UniboSviCoop", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/VectorDb/multiple_choice/questions/SviluppoCooperazioneInternazionale/SviluppoCooperazioneInternazionale.csv"),
        ("UniboMat", "https://raw.githubusercontent.com/NicolasAmadori/Tesi/refs/heads/main/VectorDb/multiple_choice/questions/matematica/matematica.csv")
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
        text_start="<s>[INST]",
        text_end="[/INST]",
        user_start="\n\n",
        system_start = "\n\n"
    )
    
    TESTING_MODEL_DICT = {
        # "microsoft/Phi-3.5-mini-instruct":phi3_5_prompt_template,
        # "meta-llama/Meta-Llama-3.1-8B-Instruct":llama3_1_prompt_template,
        "mistralai/Mistral-7B-Instruct-v0.3":mistral0_3_prompt_template
    }

    generate_answers(COLLECTION_TUPLES, TESTING_MODEL_DICT, HF_TOKEN, k=4)

def debug():
    #Testing parameters
    HF_TOKEN = "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"
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3" #Default: "jinaai/jina-embeddings-v3"
    COLLECTION_NAME = "UniboIngScInf"
    LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    mistral0_3_prompt_template = ModelPromptTemplate(
        text_start="<s>[INST]",
        text_end="[/INST]",
        user_start="\n\n",
        system_start = "\n\n"
    )

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"})

    llm = get_hugging_face_model(model_id=LLM_MODEL_NAME, hf_token = HF_TOKEN) #Download model

    chain = get_chain(llm, mistral0_3_prompt_template)

    question= "Qual è la durata del tirocinio curriculare?"
    option_a= "A. 150 ore"
    option_b= "B. 3 mesi"
    option_c= "C. 6 mesi"
    option_d= "D. 9 mesi"

    while True:
        try:
            input("Premi per generare")
            response = chain.invoke(f"{question}\n\n{option_a}\n{option_b}\n{option_c}\n{option_d}")
            logger.info(response)
        except Exception as e:
            logger.error(f"Error in debug loop: {str(e)}", exc_info=True)

if __name__ == "__main__":
    DEBUGGING = False
    if not DEBUGGING:
        main()
    else:
        debug()
