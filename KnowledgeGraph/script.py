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

import json_repair
import json
import time

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

def generateGraphWithLLM(documents):
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
        output_directory = "cleaned_documents"
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
        return True
    except Exception as e:
        print(f"Errore durante la verifica della connessione al db Neo4j: {e}")
        return False

def save_graph_to_file(generated_graph, file_name=None):
    import datetime
    file_path = f"generated_graphs"
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

def generateGraph(llm, text, create_files=False):
    if create_files:
        output_directory = "generated_graphs"
        os.makedirs(output_directory, exist_ok=True)

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
    data = json_repair.loads(llm_output)
    if create_files:
            output_path = os.path.join(output_directory, str(time.time()) + ".json")
            with open(output_path, "w") as file:
                file.write(json.dumps(data))
    return data

def convertJsonToNodes(json_array):
    from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

    nodes_dict = {}
    relationships = []

    for i, rel in enumerate(json_array):
        print(i)
        if not isinstance(rel, dict) or "head" not in rel or "head_type" not in rel or "tail" not in rel or "tail_type" not in rel:
            print(f"This relationship is not correct: {rel}")
            continue

        if isinstance(rel["head"], list) or isinstance(rel["head_type"], list) or isinstance(rel["tail"], list) or isinstance(rel["tail_type"], list):
            continue

        source_key = (rel["head"], rel["head_type"])
        if source_key not in nodes_dict:
            nodes_dict[source_key] = Node(id=rel["head"], type=rel["head_type"])
        source_node = nodes_dict[source_key]

        target_key = (rel["tail"], rel["tail_type"])
        if target_key not in nodes_dict:
            nodes_dict[target_key] = Node(id=rel["tail"], type=rel["tail_type"])
        target_node = nodes_dict[target_key]

        relationships.append(
            Relationship(
                source=source_node, target=target_node, type=rel["relation"]
            )
        )

    return GraphDocument(
        nodes=list(nodes_dict.values()),
        relationships=relationships,
        source=Document(
            page_content="Test",
            metadata={"source": "Test"}
        )
    )

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

def readJsonsFile(path):
    import json
    vettore_json = []

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)

        if os.path.isfile(filepath) and filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                vettore_json.append(data)

    return vettore_json
    
def flatAndFilterJson(input_list):
    flat_jsons = []
    for el in input_list:
        if isinstance(el, list):
            flat_jsons.extend(flatAndFilterJson(el))
        else:
            if not isinstance(el, dict) or "head" not in el or "head_type" not in el or "tail" not in el or "tail_type" not in el:
                continue
            else:
                if el["head"] == None or el["head_type"] == None or el["tail"] == None or el["tail_type"] == None:
                    continue
                if el["head"] == "" or el["head_type"] == "" or el["tail"] == "" or el["tail_type"] == "":
                    continue
                if el["head"] == "null" or el["head_type"] == "null" or el["tail"] == "null" or el["tail_type"] == "null":
                    continue
                flat_jsons.append(el)
    return flat_jsons

if __name__ == '__main__':
    # watch("neo4j")

    torch.cuda.empty_cache()
    gc.collect()

    HF_TOKEN = "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"
    os.environ["NEO4J_URI"]="neo4j+s://bbef2ff2.databases.neo4j.io"
    os.environ["NEO4J_USERNAME"]="neo4j"
    os.environ["NEO4J_PASSWORD"]="fdZslu0qGuZhCiR9pasipKRR-iLDgz9AMp8KVS9Uf2s"
    os.environ["AURA_INSTANCEID"]="bbef2ff2"
    os.environ["AURA_INSTANCENAME"]="Instance01"

    # step = 1

    # llm = getHuggingFaceModel(model_id="meta-llama/Meta-Llama-3-8B-Instruct", hf_token = HF_TOKEN)
    # print(f"\nStep #{step} Model downloaded.\n"); step+=1

    # llm_transformer = LLMGraphTransformer(llm=llm)
    # print(f"\nStep #{step} Graph Transformer initialized.\n"); step+=1

    # documents = getDocuments("/workspace/crawl")[150:250]
    # print(f"\nStep #{step} Documents read.\n"); step+=1

    # cleaned_documents = cleanDocuments(llm, documents, create_files=True)
    # print(f"\nStep #{step} Documents cleaned.\n"); step+=1
    
    # graph_json = [generateGraph(llm, d.page_content, create_files=True) for d in cleaned_documents]
    # print(f"\nStep #{step} Graph Generated.\n"); step+=1

    # jsons = flatAndFilterJson(graph_json)
    # print(f"\nStep #{step} Json array filtered and flattened. \n"); step+=1

    # graph_document = convertJsonToNodes(jsons)
    # print(f"\nStep #{step} JSON graphs converted to GraphDocument\n"); step+=1

    # graph = Neo4jGraph(url= "neo4j+s://bbef2ff2.databases.neo4j.io", username="neo4j", password="fdZslu0qGuZhCiR9pasipKRR-iLDgz9AMp8KVS9Uf2s")
    # print(f"Risultato test accesso DB Neo4j: {test_neo4j_connection(graph)}")
    # print(f"\nStep #{step} Neo4j Graph Created.\n"); step+=1
    # try:
    #     graph.add_graph_documents([graph_document])
    # except Exception as e:
    #     print(f"Errore durante l'aggiunta del grafo al db: {e}")
    # else:
    #     print(f"\nStep #{step} Graph added to neo4j db.\n"); step+=1

    ##Testing #1
    # step = 1

    # print(f"\nStep #{step}\n"); step+=1
    # llm = getHuggingFaceModel(model_id="meta-llama/Meta-Llama-3-8B-Instruct", hf_token = HF_TOKEN)

    # print(f"\nStep #{step}\n"); step+=1
    # cleaned_documents = getDocuments("/workspace/cleaned_documents")

    # print(f"\nStep #{step}\n"); step+=1
    # graphs = [generateGraph(llm, d.page_content) for d in cleaned_documents]

    # print(f"\nStep #{step}\n"); step+=1
    # graphDocument = convertJsonToNodes(graphs)

    # print(f"\nStep #{step}\n"); step+=1
    # graph = Neo4jGraph(url= "neo4j+s://bbef2ff2.databases.neo4j.io", username="neo4j", password="fdZslu0qGuZhCiR9pasipKRR-iLDgz9AMp8KVS9Uf2s")
    # save_graph_to_file(graphDocument)
    # try:
    #     graph.add_graph_documents(graphDocument)
    # except Exception as e:
    #     print(f"Errore durante l'aggiunta del grafo al db: {e}")
    # else:
    #     print("\n7. Graph added to neo4j db.\n")


    ##Testing - Load already generated jsons from files
    # step = 1

    # print(f"\nStep #{step}\n"); step+=1
    # graph_json_files = readJsonsFile("/workspace/generated_graphs")
    # print(len(graph_json_files))

    # print(f"\nStep #{step}\n"); step+=1
    # jsons = flatAndFilterJson(graph_json_files)

    # print(f"\nStep #{step}\n"); step+=1
    # graph_document = convertJsonToNodes(jsons)
    # # for n in graph_document.nodes:
    # #     # print(n)
    # #     # print(n.id == "null")
    # #     if n.id == "null" or n.id=="" or n.type=="":
    # #         print(n)
    # #         # print("############################")

    # print(f"\nStep #{step}\n"); step+=1
    # graph = Neo4jGraph(url= "neo4j+s://bbef2ff2.databases.neo4j.io", username="neo4j", password="fdZslu0qGuZhCiR9pasipKRR-iLDgz9AMp8KVS9Uf2s")
    # try:
    #     graph.add_graph_documents([graph_document])
    # except Exception as e:
    #     print(f"Errore durante l'aggiunta del grafo al db: {e} {type(e)}")
    # else:
    #     print("\n7. Graph added to neo4j db.\n")
