import faiss
import openai
import json
import numpy as np
import configparser
import pandas as pd
import sys
import os

def create_ivf_index_for_diseases(embeddings, dimension=1536, nlist=50):
    """
    질병 데이터용 IVF 인덱스를 생성합니다.
    
    Args:
        embeddings: 질병 임베딩 배열 (numpy array)
        dimension: 임베딩 차원 (기본값: 1536)
        nlist: 클러스터 수 (기본값: 50)
    
    Returns:
        훈련된 IVF 인덱스
    """
    print(f"[IVF CREATE] Creating IVF index for {len(embeddings)} diseases with {nlist} clusters...")
    print(f"[IVF CREATE] Embedding dimension: {dimension}")
    
    # IVF 인덱스 생성 (클러스터링 기반)
    quantizer = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # 인덱스 훈련
    print("[IVF CREATE] Training IVF index...")
    index.train(embeddings)
    
    # 데이터 추가
    print("[IVF CREATE] Adding vectors to IVF index...")
    index.add(embeddings)
    
    # nprobe 설정 (검색 시 탐색할 클러스터 수)
    index.nprobe = min(8, nlist)  # 질병 데이터는 상대적으로 작으므로 더 적은 클러스터 탐색
    
    print(f"[IVF CREATE] IVF index created successfully with {index.ntotal} vectors")
    print(f"[IVF CREATE] Final settings - nlist: {index.nlist}, nprobe: {index.nprobe}")
    return index

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
    
    # IVF 인덱스 사용 시 nprobe 설정 (이미 생성 시 설정되어 있음)
    # index.nprobe = min(8, index.nlist)  # 필요시 동적 조정
    
    # IVF 인덱스 정보 출력
    if hasattr(index, 'nlist'):
        print(f"[IVF INFO] Disease search - nlist: {index.nlist}, nprobe: {index.nprobe}")
    
    # FAISS 검색 (Python 바인딩)
    D, I = index.search(query_emb, top_k)
    results = [meta[i] for i in I[0]]
    return results