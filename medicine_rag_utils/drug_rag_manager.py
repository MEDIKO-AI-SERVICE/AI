import pickle
import os
import configparser
from typing import List, Dict, Any, Optional
from datetime import datetime
from .medicineRAG import recommend_drug
import faiss

class DrugRAGManager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        #keys.config 파일 경로 설정 (프로젝트 루트 기준)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, 'keys.config')
        self.config.read(config_path)
        self.openai_api_key = self.config['API_KEYS']['chatgpt_api_key']
        self.faiss_index = None
        self.meta = []
        
        #파일 경로 (medicine_rag_utils 폴더 기준)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_file = os.path.join(current_dir, "drug_data.csv")
        self.embeddings_file = os.path.join(current_dir, "drug_embeddings.pkl")
        self.meta_file = os.path.join(current_dir, "drug_data_meta.pkl")
        self.index_file = os.path.join(current_dir, "drug_data_index.index")
        self.status_file = os.path.join(current_dir, "drug_status.json")
        
        self.is_initializing = False
        self.last_update = None
        
    def initialize_rag_system(self, force_rebuild: bool = False) -> bool:
        """RAG 시스템을 초기화합니다. 이미 생성된 인덱스/메타 파일만 로드합니다."""
        if self.is_initializing:
            print("RAG system is already initializing...")
            return False
        self.is_initializing = True
        try:
            #인덱스와 메타 파일이 있으면 로드
            if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
                print("Loading existing drug index and metadata...")
                self.faiss_index = faiss.read_index(self.index_file)
                with open(self.meta_file, 'rb') as f:
                    self.meta = pickle.load(f)
                self._load_status()
                print(f"Loaded {len(self.meta)} drugs from existing index")
                self.is_initializing = False
                return True
            else:
                print(f"Index or meta file not found: {self.index_file}, {self.meta_file}")
                self.is_initializing = False
                return False
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            self.is_initializing = False
            return False
    
    def get_recommendation(self, symptom: str, language: str = "ko", patient_info: Dict = None, top_k: int = 5) -> Optional[Dict[str, Any]]:
        """증상에 따른 약품 추천을 수행합니다."""
        if not self.faiss_index or not self.meta:
            print("RAG system not initialized")
            return None
        if self.is_initializing:
            print("RAG system is still initializing")
            return None
        try:
            result = recommend_drug(
                symptom=symptom,
                faiss_index=self.faiss_index,
                meta=self.meta,
                openai_api_key=self.openai_api_key,
                language=language,
                patient_info=patient_info,
                top_k=top_k
            )
            return result
        except Exception as e:
            print(f"Error getting recommendation: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태를 반환합니다."""
        status = {
            "initialized": self.faiss_index is not None,
            "initializing": self.is_initializing,
            "drug_count": len(self.meta) if self.meta else 0,
            "csv_file_exists": os.path.exists(self.csv_file),
            "embeddings_file_exists": os.path.exists(self.embeddings_file),
            "index_file_exists": os.path.exists(self.index_file),
            "meta_file_exists": os.path.exists(self.meta_file),
            "last_update": self.last_update.isoformat() if self.last_update else None
        }
        return status
    
    def _save_status(self):
        """상태 정보를 파일에 저장합니다."""
        import json
        status = {
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "drug_count": len(self.meta) if self.meta else 0
        }
        os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(status, f)
    
    def _load_status(self):
        """상태 정보를 파일에서 로드합니다."""
        import json
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                    if status.get("last_update"):
                        self.last_update = datetime.fromisoformat(status["last_update"])
            except Exception as e:
                print(f"Error loading status: {e}")
    
    def is_ready(self) -> bool:
        """시스템이 사용 준비가 되었는지 확인합니다."""
        return self.faiss_index is not None and not self.is_initializing 