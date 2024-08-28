import os
from langchain_community.graphs import Neo4jGraph

#from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch

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

from langchain_core.prompts import PromptTemplate

from neo4j.debug import watch

def getHuggingFaceModel(model_id, hf_token):
    # model_id = "microsoft/Phi-3-mini-128k-instruct"
    # model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token = hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="cuda",
        trust_remote_code = True, #Added for Phi-3-mini-128k
        use_auth_token = hf_token
        #attn_implementation="flash_attention_2", # if you have an ampere GPU (RTX3090 OK, T4(Colab) NON OK)
    )
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=1024,
                    top_k=50,
                    temperature=0.1,
                    return_full_text = False)
    llm = HuggingFacePipeline(pipeline=pipe,
                            pipeline_kwargs={"return_full_text": False}) # <----- IMPORTANT !!!
    return llm

def getDocuments(path):
    if not os.path.exists(path):
        print("Path do not exist")
        return []
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
                        "title": content.split("\n")[2],
                        "file_name": file}))
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

def cleanMarkup(llm, text):
    template = """
    You are given a markup text. Your task is to remove any unnecessary or non-informative parts, such as:
    - Tags, unless they contain useful content.
    - Repeated phrases or sections.
    - Decorative characters or symbols.
    - Empty lines or spaces.

    Please leave informative links.

    Output only the cleaned text, without any additional explanation or markup.

    {text}
    """

    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm

    return chain.invoke({"text": text})   

def cleanDocuments(llm, documents, create_files=False):
    if create_files:
        output_directory = "workspace/cleaned_documents"
        os.makedirs(output_directory, exist_ok=True)
    i = 0
    for d in documents:
        i+=1
        print(f"Cleaning the document number: {i}/{len(documents)}")
        d.page_content = cleanMarkup(llm, d.page_content)
        if create_files:
            output_path = os.path.join(output_directory, d.metadata['file_name'])
            with open(output_path, "w") as file:
                file.write(d.page_content)
    return documents

def test_neo4j_connection(graph):
    try:
        result = graph.query("RETURN 'Connection successful' AS message")
        print(result)
        return True
    except Exception as e:
        print(f"Errore durante la verifica della connessione al db Neo4j: {e}")
        return False

def save_graph_to_file(generated_graph, file_name=None):
    file_path = f"workspace/generated_graphs"
    if file_name:
        file_path += f"/{file_name}.txt"
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path += f"/generated_graph_{current_time}.txt"

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for graph_document in generated_graph:
                f.write(str(graph_document) + '\n')
        print(f"Graph successfully saved to {file_path}")
    except Exception as e:
        print(f"Error while saving the graph to file: {e}")

def generateGraph(llm, text):
    import json_repair

    template = """
    You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
    Your task is to identify the entities and relations requested with the user prompt from a given text.
    You must generate the output in a JSON format containing a list 'with JSON objects.
    Each object should have the keys: head, ''head_type, relation, tail, and tail_type.
    The head 'key must contain the text of the extracted entity with one of the types from the provided list in the user prompt.
    Attempt to extract as many entities and relations as you can.
    Maintain Entity Consistency: When extracting entities, it's vital to ensure 'consistency.
    If an entity, such as John Doe, is mentioned multiple 'times in the text but is referred to by different names or pronouns '(e.g., Joe, he), always use the most complete identifier for 'that entity.
    The knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.

    IMPORTANT NOTES:\n- Don't add any explanation and text.

    {text}
    """

    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm
    llm_output = chain.invoke({"text": text})  
    return json_repair.loads(llm_output)

def scrivi_json_in_file(json_data, percorso):
    import json
    if not os.path.exists(percorso):
        os.makedirs(percorso)
    
    for i, data in enumerate(json_data):
        nome_file = f"json_{i+1}.json"
        percorso_completo = os.path.join(percorso, nome_file)
        
        with open(percorso_completo, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

def saveTextToFiles(texts, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i, testo in enumerate(texts):
        nome_file = f"text_{i+1}.txt"
        percorso_completo = os.path.join(path, nome_file)
        with open(percorso_completo, 'w', encoding='utf-8') as file:
            file.write(testo)

if __name__ == '__main__':
    watch("neo4j")

    torch.cuda.empty_cache()
    gc.collect()

    HF_TOKEN = "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"
    os.environ["NEO4J_URI"]="neo4j+s://bbef2ff2.databases.neo4j.io"
    os.environ["NEO4J_USERNAME"]="neo4j"
    os.environ["NEO4J_PASSWORD"]="fdZslu0qGuZhCiR9pasipKRR-iLDgz9AMp8KVS9Uf2s"
    os.environ["AURA_INSTANCEID"]="bbef2ff2"
    os.environ["AURA_INSTANCENAME"]="Instance01"

    # graph = Neo4jGraph(url= "neo4j+s://bbef2ff2.databases.neo4j.io", username="neo4j", password="fdZslu0qGuZhCiR9pasipKRR-iLDgz9AMp8KVS9Uf2s")
    # print("\n1. Neo4j Graph Created.\n")
    # print(f"Risultato test: {test_neo4j_connection(graph)}")

    llm = getHuggingFaceModel(model_id="meta-llama/Meta-Llama-3-8B-Instruct", hf_token = HF_TOKEN)
    print("\n2. Model downloaded.\n")

    llm_transformer = LLMGraphTransformer(llm=llm)
    print("\n3. Graph Transformer initialized.\n")

    # documents = getDocuments("/workspace/crawl")
    # print(f"\n4. {len(documents)} Documents read.\n")

    # cleaned_documents = cleanDocuments(llm, documents, create_files=True)
    # print(f"\n5. Documents cleaned.\n")
    
    # generated_graph = generateGraph(cleaned_documents)
    # print("\n6. Graph Generated.\n")

    # save_graph_to_file(generated_graph)
    # try:
    #     graph.add_graph_documents(generated_graph)
    # except Exception as e:
    #     print(f"Errore durante l'aggiunta del grafo al db: {e}")
    # else:
    #     print("\n7. Graph added to neo4j db.\n")

    ##Testing

    cleaned_documents = getDocuments("/workspace/cleaned_documents")
    graphs = [generateGraph(llm, d.page_content) for d in cleaned_documents]
    scrivi_json_in_file(graphs, "workspace/generated_graphs")