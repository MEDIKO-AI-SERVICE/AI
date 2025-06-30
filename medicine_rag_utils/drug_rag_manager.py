import pickle
import os
import configparser
from typing import List, Dict, Any, Optional
from datetime import datetime
from .drug_api_client import DrugAPIClient
from .vector_faiss import build_faiss_index
from .medicineRAG import recommend_drug

class DrugRAGManager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('keys.config')
        self.openai_api_key = self.config['API_KEYS']['chatgpt_api_key']
        self.faiss_index = None
        self.meta = []
        self.index_file = "medicine_rag_utils/drug_index.pkl"
        self.meta_file = "medicine_rag_utils/drug_meta.pkl"
        self.status_file = "medicine_rag_utils/drug_status.json"
        self.is_initializing = False
        self.last_update = None
        
    def initialize_rag_system(self, force_rebuild: bool = False) -> bool:
        """RAG 시스템을 초기화합니다."""
        if self.is_initializing:
            print("RAG system is already initializing...")
            return False
            
        self.is_initializing = True
        
        try:
            # 기존 인덱스가 있고 강제 재구축이 아닌 경우 로드
            if not force_rebuild and os.path.exists(self.index_file) and os.path.exists(self.meta_file):
                print("Loading existing drug index and metadata...")
                with open(self.index_file, 'rb') as f:
                    self.faiss_index = pickle.load(f)
                with open(self.meta_file, 'rb') as f:
                    self.meta = pickle.load(f)
                
                # 상태 정보 로드
                self._load_status()
                
                print(f"Loaded {len(self.meta)} drugs from existing index")
                self.is_initializing = False
                return True
            
            # 새로운 데이터 수집 및 인덱스 구축
            print("Collecting drug data from API...")
            api_client = DrugAPIClient()
            drug_data = api_client.get_all_drugs()
            
            if not drug_data:
                print("No drug data collected")
                self.is_initializing = False
                return False
            
            print(f"Building FAISS index for {len(drug_data)} drugs...")
            self.faiss_index, self.meta = build_faiss_index(drug_data, self.openai_api_key)
            
            # 인덱스와 메타데이터 저장
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.faiss_index, f)
            with open(self.meta_file, 'wb') as f:
                pickle.dump(self.meta, f)
            
            # 상태 정보 저장
            self.last_update = datetime.now()
            self._save_status()
            
            print(f"Successfully built and saved index for {len(self.meta)} drugs")
            self.is_initializing = False
            return True
            
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