import faiss
import openai
import json
import numpy as np
import configparser
import pandas as pd
import sys
import os

def get_embedding(text, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def load_index_and_meta(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return index, meta

def search_similar_diseases(symptom_list, index_path, meta_path, top_k=3):
    config = configparser.ConfigParser()
    # keys.config 파일 경로 설정 (프로젝트 루트 기준)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'keys.config')
    config.read(config_path)
    openai_api_key = config['API_KEYS']['chatgpt_api_key']
    text = ', '.join(symptom_list)
    query_emb = get_embedding(text, openai_api_key).reshape(1, -1)
    index, meta = load_index_and_meta(index_path, meta_path)
    # FAISS 검색 (Python 바인딩)
    D, I = index.search(query_emb, top_k)
    results = [meta[i] for i in I[0]]
    return results