import os
from langchain_community.graphs import Neo4jGraph

#from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
from google.colab import userdata

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from langchain_huggingface import HuggingFacePipeline

from langchain_core.documents import Document

import gc

from llm import LLMGraphTransformer

def getHuggingFaceModel(modelId = "microsoft/Phi-3-mini-128k-instruct"):
    # model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(modelId)

    model = AutoModelForCausalLM.from_pretrained(
        modelId,
        load_in_4bit=True,
        device_map="cuda",
        trust_remote_code = True #Added for Phi-3-mini-128k
        #attn_implementation="flash_attention_2", # if you have an ampere GPU (RTX3090 OK, T4(Colab) NON OK)
    )
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=1024,
                    top_k=50,
                    temperature=0.1)
    llm = HuggingFacePipeline(pipeline=pipe,
                            pipeline_kwargs={"return_full_text": False}) # <----- IMPORTANT !!!
    return llm

def getDocuments(path):
    # text = """
    # Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    # She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    # Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
    # She was, in 1906, the first woman to become a professor at the University of Paris.
    # """
    # documents = [Document(page_content=text)]

    documents = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            content = ""
            try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            except Exception as e:
            print(f"Errore nella lettura di {file_path}: {e}")

            documents.append(Document(
                page_content=content,
                metadata={"path": file_path,
                        "title": content.split("\n")[2]}))
    return documents

def generateGraph(documents):
    gc.collect()
    torch.cuda.empty_cache()
    i = 0
    total_graph_documents = []
    for document in documents:
        i+=1
        print(f"Converting the document number: {i}/{len(documents)}")
        graph_documents = llm_transformer.convert_to_graph_documents([document])
        total_graph_documents.extend(graph_documents)
        del graph_documents
        #Cleaning the space
        torch.cuda.empty_cache()
        if(i % 3 == 0):
            gc.collect()
    
    return total_graph_documents

if __name__ == '__main__':
    os.environ["NEO4J_URI"]="neo4j+s://6743dae3.databases.neo4j.io"
    os.environ["NEO4J_USERNAME"]="neo4j"
    os.environ["NEO4J_PASSWORD"]="gUxckYtBrObmia7f1ByzOp_0H4GPlW6-wZha7TofvEI"
    os.environ["AURA_INSTANCEID"]="6743dae3"
    os.environ["AURA_INSTANCENAME"]="Instance01"
    
    graph = Neo4jGraph()

    llm = getHuggingFaceModel()
    llm_transformer = LLMGraphTransformer(llm=llm)

    documents = getDocuments("path")
    generated_graph = generateGraph(documents[:5])
    graph.add_graph_documents(generated_graph)