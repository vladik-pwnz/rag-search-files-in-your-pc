import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from qdrant_client import QdrantClient

app = FastAPI()

# Параметры
COLL_NAME = 'documents'
ENCODER_NAME = 'intfloat/multilingual-e5-large'
VEC_SIZE = 1024  # Размер вектора энкодера
CHUNK_SIZE = 1430
CHUNK_OVERLAP = 138
FILES_DIR = 'files'

# Модель для кодирования
bi_encoder = SentenceTransformer(ENCODER_NAME)

# Модель для генерации ответов
model_id = 'microsoft/Phi-3-mini-4k-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Qdrant клиент
qdrant_client = QdrantClient("http://localhost:6333")

class SearchRequest(BaseModel):
    query: str

class DocumentSummary(BaseModel):
    folder: str
    filename: str
    summary: str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def str_to_vec(encoder, text):
    # Пример преобразования строки в вектор с использованием энкодера
    return encoder.encode(text)

def vec_search(encoder, query, n_top_cos=5):
    query_emb = str_to_vec(encoder, query)
    search_result = qdrant_client.search(
        collection_name=COLL_NAME,
        query_vector=query_emb,
        limit=n_top_cos,
        with_vectors=False
    )
    top_chunks = [x.payload['chunk'] for x in search_result]
    top_files = list(set([x.payload['file'] for x in search_result]))
    return top_chunks, top_files

def get_llm_answer(query, chunks_join, max_new_tokens=100, temperature=0.7, top_p=0.9, top_k=50):
    user_prompt = '''Используй только следующий контекст, чтобы очень кратко ответить на вопрос в конце.
    Не пытайся выдумывать ответ.
    Контекст:
    ===========
    {chunks_join}
    ===========
    Вопрос:
    ===========
    {query}'''.format(chunks_join=chunks_join, query=query)
    
    SYSTEM_PROMPT = "Don't write instructions, any user or tutor possible future sentences! Ты русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. Не пиши инструкции! Отвечай только на русском языке!"
    RESPONSE_TEMPLATE = "assistant\n"
    
    prompt = f'''system\n{SYSTEM_PROMPT}\nuser\n{user_prompt}\n{RESPONSE_TEMPLATE}'''
    
    def generate(model, tokenizer, prompt):
        data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(model.device) for k, v in data.items()}
        output_ids = model.generate(
            **data,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            no_repeat_ngram_size=15,
            repetition_penalty=1.1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p 
        )[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        return output.strip()
    
    response = generate(model, tokenizer, prompt)
    logger.info(f"LLM response: {response}")
    return response

@app.post("/search", response_model=List[DocumentSummary])
async def search_files(request: SearchRequest):
    logger.info(f"Received search query: {request.query}")
    query = request.query
    top_chunks, top_files = vec_search(bi_encoder, query)
    chunks_join = " ".join(top_chunks)
    response = get_llm_answer(query, chunks_join)
    results = [{"folder": FILES_DIR, "filename": f, "summary": response} for f in top_files]
    logger.info(f"Returning results: {results}")
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
