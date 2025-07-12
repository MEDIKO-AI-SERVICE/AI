#!/usr/bin/env python3
"""
약품 RAG용 IVF 인덱스 생성 스크립트
기존 임베딩을 사용하여 IVF 인덱스를 생성합니다.
"""

import os
import sys
import pickle
import faiss
import numpy as np
from medicineRAG import create_ivf_index_for_drugs

def create_drug_ivf_index():
    """약품 데이터용 IVF 인덱스를 생성합니다."""
    
    # 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_file = os.path.join(current_dir, "drug_embeddings.pkl")
    meta_file = os.path.join(current_dir, "drug_data_meta.pkl")
    ivf_index_file = os.path.join(current_dir, "drug_data_ivf_index.index")
    
    # 기존 임베딩 파일 확인
    if not os.path.exists(embeddings_file):
        print(f"Embeddings file not found: {embeddings_file}")
        print("Note: You need to create embeddings first from the drug data")
        return False
    
    try:
        # 기존 임베딩 로드
        print("Loading drug embeddings...")
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        print(f"Loaded {len(embeddings)} drug embeddings with shape {embeddings.shape}")
        
        # 데이터 크기에 맞게 nlist 조정
        nlist = min(100, max(10, len(embeddings) // 10))
        
        # IVF 인덱스 생성
        ivf_index = create_ivf_index_for_drugs(
            embeddings, 
            dimension=embeddings.shape[1], 
            nlist=nlist
        )
        
        # IVF 인덱스 저장
        faiss.write_index(ivf_index, ivf_index_file)
        print(f"IVF index saved to {ivf_index_file}")
        
        return True
        
    except Exception as e:
        print(f"Error creating drug IVF index: {e}")
        return False

if __name__ == "__main__":
    print("Creating drug IVF index...")
    success = create_drug_ivf_index()
    if success:
        print("Drug IVF index created successfully!")
    else:
        print("Failed to create drug IVF index")
        sys.exit(1) 