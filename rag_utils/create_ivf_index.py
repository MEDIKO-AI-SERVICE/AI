#!/usr/bin/env python3
"""
질병 RAG용 IVF 인덱스 생성 스크립트
기존 임베딩을 사용하여 IVF 인덱스를 생성합니다.
"""

import os
import sys
import pickle
import faiss
import numpy as np
from rag_search import create_ivf_index_for_diseases

def create_disease_ivf_index():
    """질병 데이터용 IVF 인덱스를 생성합니다."""
    
    # 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_file = os.path.join(current_dir, "disease_embeddings.pkl")
    meta_file = os.path.join(current_dir, "combined_meta.json")
    ivf_index_file = os.path.join(current_dir, "combined_ivf.index")
    
    # 기존 임베딩 파일 확인
    if not os.path.exists(embeddings_file):
        print(f"Embeddings file not found: {embeddings_file}")
        print("Note: You need to create embeddings first from the disease data")
        return False
    
    try:
        # 기존 임베딩 로드
        print("Loading disease embeddings...")
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        print(f"Loaded {len(embeddings)} disease embeddings with shape {embeddings.shape}")
        
        # 데이터 크기에 맞게 nlist 조정
        nlist = min(50, max(5, len(embeddings) // 20))
        
        # IVF 인덱스 생성
        ivf_index = create_ivf_index_for_diseases(
            embeddings, 
            dimension=embeddings.shape[1], 
            nlist=nlist
        )
        
        # IVF 인덱스 저장
        faiss.write_index(ivf_index, ivf_index_file)
        print(f"IVF index saved to {ivf_index_file}")
        
        return True
        
    except Exception as e:
        print(f"Error creating disease IVF index: {e}")
        return False

if __name__ == "__main__":
    print("Creating disease IVF index...")
    success = create_disease_ivf_index()
    if success:
        print("Disease IVF index created successfully!")
    else:
        print("Failed to create disease IVF index")
        sys.exit(1) 