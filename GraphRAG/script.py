from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from neo4j import GraphDatabase
# from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from dotenv import load_dotenv

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from llm import LLMGraphTransformer

class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

def showGraph():
    driver = GraphDatabase.driver(
        uri = os.environ["NEO4J_URI"],
        auth = (os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"]))
    session = driver.session()
    widget = GraphWidget(graph = session.run("MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t").graph())
    widget.node_label_mapping = 'id'
    return widget

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()

# Fulltext index query
def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def full_retriever(question: str):
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    final_data = f"""Graph data:
    {graph_data}
    vector data:
    {"#Document ". join(vector_data)}
        """
    return final_data
    
def getHuggingFaceModel(model_id, hf_token):
    # model_id = "microsoft/Phi-3-mini-128k-instruct"
    # model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="cuda",
        trust_remote_code = True, #Added for Phi-3-mini-128k
        token = hf_token
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

if __name__ == '__main__':
    HF_TOKEN = "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"

    load_dotenv()

    # ################################################################################################
    loader = TextLoader(file_path="dummytext.txt")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
    documents = text_splitter.split_documents(documents=docs)

    # ################################################################################################
    # llm_type = os.getenv("LLM_TYPE", "ollama")
    # if llm_type == "ollama":
    #     llm = ChatOllama(model="llama3.1", temperature=0)
    # else:
    #     llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    llm = getHuggingFaceModel(model_id="meta-llama/Meta-Llama-3-8B-Instruct", hf_token = os.getenv("HF_TOKEN"))
    llm_transformer = LLMGraphTransformer(llm=llm)

    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # ################################################################################################
    # print(graph_documents[0])

    # ################################################################################################
    graph = Neo4jGraph(url= "neo4j+s://bbef2ff2.databases.neo4j.io", username="neo4j", password="fdZslu0qGuZhCiR9pasipKRR-iLDgz9AMp8KVS9Uf2s")
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

    # ################################################################################################
    # showGraph()

    # ################################################################################################
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    vector_retriever = vector_index.as_retriever()

    # ################################################################################################
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )

    entity_chain = prompt | llm.with_structured_output(Entities)

    # ################################################################################################
    # entity_chain.invoke({"question": "Who are Nonna Lucia and Giovanni Caruso?"}).names

    # ################################################################################################
    # print(graph_retriever("Who is Nonna Lucia?"))

    # ################################################################################################
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {
                "context": full_retriever,
                "question": RunnablePassthrough(),
            }
        | prompt
        | llm
        | StrOutputParser()
    )

    chain.invoke(input="Who is Nonna Lucia? Did she teach anyone about restaurants or cooking?")