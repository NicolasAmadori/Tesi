import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_core.documents import Document
import gc
import torch

import re
from typing import List

#vllm
from vllm import SamplingParams
from vllm import LLM
# from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

def create_prompt(text:str, tokenizer) -> str:
   messages = [
    {
        "role": "user",
        "content": text
    },
   ]
   return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
   )

def getDocuments(path) -> List[Document]:
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
                        "file_name": file}))
    return documents

def clearMemory():
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor.driver_worker
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    print(f"cuda memory: {torch.cuda.memory_allocated() // 1024 // 1024}MB")

def get_sampling_params():
    top_k = 1 # @param {type:"integer"}
    temperature = 0 # @param {type:"slider", min:0, max:1, step:0.1}
    repetition_penalty = 1.08 # @param {type:"number"}
    presence_penalty = 0.25 # @param {type:"slider", min:0, max:1, step:0.1}
    top_k = 1 # @param {type:"integer"}
    max_tokens = 4096 # @param {type:"integer"}
    sampling_params = SamplingParams(temperature=temperature, top_k=top_k, presence_penalty=presence_penalty, repetition_penalty=repetition_penalty, max_tokens=max_tokens)
    return sampling_params
    
def cleanDocuments(llm: LLM,
    sampling_params: SamplingParams,
    documents: List[Document],
    create_files: bool = True,
    output_folder_name: str = "cleaned_documents") -> List[Document]:

    cleaned_documents = []
    for d in documents:
        html = d.page_content
        prompt = create_prompt(html, llm.get_tokenizer())
        results = llm.generate(prompt, sampling_params=sampling_params)
        generated_text = results[0].outputs[0].text
        cleaned_documents.append(Document(
                page_content=generated_text,
                metadata={"path": d.metadata["path"],
                        "file_name": d.metadata["file_name"]}))

        if create_files:
            output_path = os.path.join(output_folder_name, d.metadata['path'])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as file:
                file.write(generated_text)
    return cleaned_documents

def main():
    torch.cuda.empty_cache()
    # model_id = "jinaai/reader-lm-1.5b"

    # device = "cuda"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    #     device_map=device,
    #     trust_remote_code = True,
    # )
    
    documents = getDocuments("IngegneriaScienzeInformatiche")
    print(len(documents))

    sampling_params = get_sampling_params()
    llm = LLM(model='jinaai/reader-lm-1.5b', dtype='float16', max_model_len=48000, gpu_memory_utilization=0.9)

    cleaned_documents = cleanDocuments(llm, sampling_params, documents)
    # clearMemory()
    
if __name__ == '__main__':
    main()