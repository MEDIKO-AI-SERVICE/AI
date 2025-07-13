import openai
import numpy as np
import faiss
from typing import List, Dict, Any
from rag_utils.rag_search import get_embedding
from gpt_utils.prompting_gpt import translate_text, romanize_korean_names, recommend_drug_llm_response
import re
import time

def create_ivf_index_for_drugs(embeddings, dimension=1536, nlist=100):
    """
    약품 데이터용 IVF 인덱스를 생성합니다.
    
    Args:
        embeddings: 약품 임베딩 배열 (numpy array)
        dimension: 임베딩 차원 (기본값: 1536)
        nlist: 클러스터 수 (기본값: 100)
    
    Returns:
        훈련된 IVF 인덱스
    """
    print(f"[IVF CREATE] Creating IVF index for {len(embeddings)} drugs with {nlist} clusters...")
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
    index.nprobe = min(10, nlist)  # 검색 정확도와 속도 균형
    
    print(f"[IVF CREATE] IVF index created successfully with {index.ntotal} vectors")
    print(f"[IVF CREATE] Final settings - nlist: {index.nlist}, nprobe: {index.nprobe}")
    return index

def recommend_drug(symptom: str, faiss_index, meta: List[Dict], openai_api_key: str, language: str = "ko", patient_info: Dict = None, top_k: int = 3) -> Dict[str, Any]:
    """
    증상에 따른 약품 추천과 약사에게 할 질문을 생성합니다.
    
    Args:
        symptom: 사용자의 증상 (다국어 가능)
        faiss_index: FAISS 인덱스
        meta: 약품 메타데이터
        openai_api_key: OpenAI API 키
        language: 언어 코드 (ko, en, vi, zh_cn, zh_tw, ne, id, th)
        patient_info: 환자 정보 (성별, 나이, 키, 몸무게, 가족력, 알레르기, 현재복용약, 과거병력)
        top_k: 초기 검색할 약품 수
    
    Returns:
        추천 결과와 질문 리스트
    """
    start_total = time.time()
    t0 = time.time()
    #1단계: 증상을 한국어로 변환 (RAG 검색을 위해)
    korean_symptom = convert_symptom_to_korean(symptom, language)
    print(f"[latency][RAG] 증상 번역: {time.time() - t0:.3f}초", flush=True)
    t0 = time.time()
    #2단계: 환자 정보를 고려한 검색 범위 확장
    #환자 정보가 있으면 더 많은 후보를 검색하여 필터링
    search_k = top_k * 2 if patient_info else top_k
    
    #한국어 증상으로 RAG 검색
    query_emb = get_embedding(korean_symptom, openai_api_key).reshape(1, -1)
    print(f"[latency][RAG] 임베딩 생성: {time.time() - t0:.3f}초", flush=True)
    t0 = time.time()
    
    # IVF 인덱스 사용 시 nprobe 설정 (이미 생성 시 설정되어 있음)
    # faiss_index.nprobe = min(10, faiss_index.nlist)  # 필요시 동적 조정
    
    # IVF 인덱스 정보 출력
    if hasattr(faiss_index, 'nlist'):
        print(f"[IVF INFO] Drug search - nlist: {faiss_index.nlist}, nprobe: {faiss_index.nprobe}")
    
    #FAISS 검색 (Python 바인딩)
    D, I = faiss_index.search(query_emb, search_k)
    print(f"[latency][RAG] FAISS 검색: {time.time() - t0:.3f}초", flush=True)
    t0 = time.time()
    #후보 약품들 선택
    candidates = [meta[i] for i in I[0]]
    
    #3단계: 환자 정보를 고려한 약품 필터링 및 순위 조정
    filtered_candidates = filter_and_rank_drugs(candidates, patient_info, korean_symptom)
    print(f"[latency][RAG] 후보 필터링/랭킹: {time.time() - t0:.3f}초", flush=True)
    t0 = time.time()
    #4단계: 최적 약품 후보 top-N 전달 (없으면 에러)
    if not filtered_candidates:
        print(f"[latency][RAG] 전체 소요 시간(에러): {time.time() - start_total:.3f}초", flush=True)
        return create_error_response("적합한 약품을 찾을 수 없습니다.", language)
    
    top_candidates = filtered_candidates[:top_k]  # LLM에 넘길 후보 수도 top_k로 통일
    
    #5단계: LLM에 모든 결과를 한 번에 요청
    result = recommend_drug_llm_response(top_candidates, symptom, patient_info)
    print(f"[latency][RAG] LLM 호출: {time.time() - t0:.3f}초", flush=True)
    print(f"[latency][RAG] 전체 소요 시간: {time.time() - start_total:.3f}초", flush=True)
    
    return result

def filter_and_rank_drugs(candidates: List[Dict], patient_info: Dict, symptom: str) -> List[Dict]:
    """환자 정보를 고려하여 약품을 필터링하고 순위를 조정합니다."""
    if not patient_info:
        return candidates
    
    # 키워드 기반 빠른 필터링 (GPT 호출 없이)
    scored_candidates = []
    
    for drug in candidates:
        #기본 점수 (증상 매칭)
        score = 10
        warnings = []
        
        #키워드 기반 분석 (빠른 필터링)
        keyword_score, keyword_warnings = analyze_drug_keywords(drug, patient_info)
        score += keyword_score
        warnings.extend(keyword_warnings)
        
        scored_candidates.append({
            'drug': drug,
            'score': score,
            'warnings': warnings
        })
    
    #점수순으로 정렬 (높은 점수 우선)
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    #경고가 있는 약품은 후순위로
    safe_candidates = [c for c in scored_candidates if not c['warnings']]
    warning_candidates = [c for c in scored_candidates if c['warnings']]
    
    #안전한 약품 우선, 그 다음 경고가 있는 약품
    final_candidates = safe_candidates + warning_candidates
    
    return [c['drug'] for c in final_candidates]

def analyze_drug_context_batch(drugs: List[Dict], patient_info: Dict, symptom: str) -> List[tuple[int, List[str]]]:
    """GPT를 활용한 배치 맥락 기반 약품 분석"""
    try:
        #환자 정보 구성
        patient_context = f"""
        증상: {symptom}
        성별: {patient_info.get('gender', 'N/A')}
        나이: {patient_info.get('age', 'N/A')}
        키: {patient_info.get('height', 'N/A')}cm
        몸무게: {patient_info.get('weight', 'N/A')}kg
        알레르기: {patient_info.get('allergies', 'N/A')}
        현재복용약: {patient_info.get('current_medications', 'N/A')}
        과거병력: {patient_info.get('medical_history', 'N/A')}
        """
        
        # 모든 약품 정보를 하나의 프롬프트로 구성
        drug_infos = []
        for i, drug in enumerate(drugs):
            drug_info = f"""
            약품 {i+1}:
            약품명: {drug.get('itemName', '')}
            효능: {drug.get('efcyQesitm', '')}
            복용법: {drug.get('useMethodQesitm', '')}
            주의사항: {drug.get('atpnQesitm', '')}
            부작용: {drug.get('seQesitm', '')}
            상호작용: {drug.get('intrcQesitm', '')}
            """
            drug_infos.append(drug_info)
        
        all_drug_info = "\n".join(drug_infos)
        
        prompt = f"""당신은 의학 전문가입니다. 다음 환자 정보와 약품 정보들을 분석하여 각 약품의 적합성을 평가해주세요.

환자 정보:
{patient_context}

약품 정보:
{all_drug_info}

각 약품에 대해 다음 기준으로 평가해주세요:
1. 증상과 약품 효능의 적합성 (0-5점)
2. 환자 특성(나이, 성별)과 약품의 적합성 (0-3점)
3. 안전성 (알레르기, 상호작용, 부작용 고려) (0-5점)
4. 복용 편의성 (나이, 신체 조건 고려) (0-2점)

JSON 배열 형태로 응답해주세요:
[
    {{
        "drug_index": 0,
        "score": 점수,
        "risks": ["위험요소1", "위험요소2"],
        "reasoning": "평가 근거"
    }},
{{
        "drug_index": 1,
    "score": 점수,
    "risks": ["위험요소1", "위험요소2"],
    "reasoning": "평가 근거"
    }}
]"""

        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        import json
        results = json.loads(response.choices[0].message.content)
        
        # 결과를 약품 순서대로 정렬
        sorted_results = sorted(results, key=lambda x: x.get('drug_index', 0))
        
        # (score, risks) 튜플 리스트 반환
        return [(r.get('score', 0), r.get('risks', [])) for r in sorted_results]
        
    except Exception as e:
        print(f"Error in batch context analysis: {e}")
        # 에러 시 기본값 반환
        return [(0, []) for _ in drugs]

def analyze_drug_keywords(drug: Dict, patient_info: Dict) -> tuple[int, List[str]]:
    """키워드 기반 약품 분석 (환자 특성 고려) - 빠른 필터링용"""
    score = 0
    warnings = []
    
    #성별 고려
    if patient_info.get('gender'):
        gender = patient_info['gender'].lower()
        drug_name = drug.get('itemName', '').lower()
        drug_purpose = drug.get('efcyQesitm', '').lower()
        drug_text = f"{drug_name} {drug_purpose}"
        
        if gender in ['female', '여성', '여자', '女', '女姓', 'đàn bà', 'nữ giới']:
            #여성 전용 약품 키워드
            female_keywords = [
                '여성', '여자', '생리', '월경', '임신', '수유', '그날엔', '우먼', 'woman',
                '여성용', '여자용', '생리통', '월경통', '생리불순', '월경불순',
                '여성호르몬', '에스트로겐', '프로게스테론', '피임', '갱년기'
            ]
            if any(keyword in drug_text for keyword in female_keywords):
                score += 8
            #남성 전용 약품은 제외
            male_keywords = ['남성', '남자', '전립선', '남성용', '남자용']
            if any(keyword in drug_text for keyword in male_keywords):
                score -= 10
                warnings.append("남성 전용 약품")
                
        elif gender in ['male', '남성', '남자', '男', '男姓', 'người đàn ông', 'nam giới']:
            #남성 전용 약품 키워드
            male_keywords = [
                '남성', '남자', '전립선', '남성용', '남자용', '발기', '정력',
                '남성호르몬', '테스토스테론', '탈모', '남성갱년기'
            ]
            if any(keyword in drug_text for keyword in male_keywords):
                score += 8
            #여성 전용 약품은 제외
            female_keywords = ['여성', '여자', '생리', '월경', '임신', '수유', '그날엔', '우먼']
            if any(keyword in drug_text for keyword in female_keywords):
                score -= 10
                warnings.append("여성 전용 약품")
    
    #나이 고려
    if patient_info.get('age'):
        age = patient_info['age']
        drug_name = drug.get('itemName', '').lower()
        drug_method = drug.get('useMethodQesitm', '').lower()
        drug_text = f"{drug_name} {drug_method}"
        
        if age < 12:  #어린이
            child_keywords = ['시럽', '츄어블', '어린이', '소아', '아동', '유아', '베이비', 'baby', '유소년', '초등학생', '유치원생']
            if any(keyword in drug_text for keyword in child_keywords):
                score += 5
            #성인 전용 약품은 제외
            adult_keywords = ['성인', '어른', '성인용', '어른용']
            if any(keyword in drug_text for keyword in adult_keywords):
                score -= 8
                warnings.append("성인 전용 약품")
                
        elif age < 18:  #청소년
            teen_keywords = ['청소년', '학생', '10대', 'teen', '중학생', '고등학생', '중학교', '고등학교', '유소년']
            if any(keyword in drug_text for keyword in teen_keywords):
                score += 3
                
        elif age > 65:  #고령자
            elderly_keywords = ['고령자', '노인', '시니어', 'elderly', 'senior']
            if any(keyword in drug_text for keyword in elderly_keywords):
                score += 5
    
    #키/몸무게 고려
    if patient_info.get('height') and patient_info.get('weight'):
        height = patient_info['height']
        weight = patient_info['weight']
        drug_text = f"{drug.get('useMethodQesitm', '')} {drug.get('atpnQesitm', '')}".lower()
        
        #BMI 계산
        bmi = weight / ((height/100) ** 2)
        if bmi > 30:  #비만
            if '비만' in drug_text or '체중' in drug_text:
                score += 2
    
    #알레르기 고려
    if patient_info.get('allergies'):
        allergies = patient_info['allergies'].lower()
        drug_text = f"{drug.get('atpnQesitm', '')} {drug.get('seQesitm', '')}".lower()
        
        for allergy in allergies.split(','):
            allergy = allergy.strip()
            if allergy in drug_text:
                score -= 10
                warnings.append(f"알레르기 반응 가능성: {allergy}")
    
    #현재 복용약 고려
    if patient_info.get('current_medications'):
        current_meds = patient_info['current_medications'].lower()
        drug_text = f"{drug.get('intrcQesitm', '')} {drug.get('atpnQesitm', '')}".lower()
        
        if any(med in drug_text for med in current_meds.split(',')):
            score -= 5
            warnings.append("현재 복용약과 상호작용 가능성")
    
    #과거 병력 고려
    if patient_info.get('medical_history'):
        history = patient_info['medical_history'].lower()
        drug_text = f"{drug.get('atpnQesitm', '')} {drug.get('seQesitm', '')}".lower()
        
        if any(condition in drug_text for condition in history.split(',')):
            score -= 3
            warnings.append("과거 병력과 관련된 주의사항")
    
    return score, warnings


def create_error_response(message: str, language: str) -> Dict[str, Any]:
    """에러 응답을 생성합니다."""
    if language.lower() != "ko":
        message = translate_text(message, language)
    
    return {
        "drug_name": "",
        "drug_purpose": "",
        "drug_image_url": "",
        "pharmacist_questions": [],
        "warning_message": message,
        "error": True
    }

def convert_symptom_to_korean(symptom: str, source_language: str) -> str:
    """증상을 한국어로 변환합니다."""
    if source_language.lower() == "ko":
        return symptom
    
    client = openai.OpenAI(api_key=openai.api_key)
    
    prompt = f"""You are a medical translator. Convert the following symptom to Korean medical terminology. Source language: {source_language}, Symptom: {symptom}. Return only the Korean translation. Do not add any explanations or additional text. Focus on medical accuracy and use proper Korean medical terms."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error converting symptom to Korean: {e}")
        return symptom  #실패 시 원본 반환