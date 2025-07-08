import openai
import numpy as np
import faiss
from typing import List, Dict, Any
from rag_utils.rag_search import get_embedding
from gpt_utils.prompting_gpt import translate_text, romanize_korean_names, recommend_drug_llm_response
import re

def recommend_drug(symptom: str, faiss_index, meta: List[Dict], openai_api_key: str, language: str = "ko", patient_info: Dict = None, top_k: int = 5) -> Dict[str, Any]:
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
    #1단계: 증상을 한국어로 변환 (RAG 검색을 위해)
    korean_symptom = convert_symptom_to_korean(symptom, language)
    
    #2단계: 환자 정보를 고려한 검색 범위 확장
    #환자 정보가 있으면 더 많은 후보를 검색하여 필터링
    search_k = top_k * 3 if patient_info else top_k
    
    #한국어 증상으로 RAG 검색
    query_emb = get_embedding(korean_symptom, openai_api_key).reshape(1, -1)
    #FAISS 검색 (Python 바인딩)
    D, I = faiss_index.search(query_emb, search_k)
    
    #후보 약품들 선택
    candidates = [meta[i] for i in I[0]]
    
    #3단계: 환자 정보를 고려한 약품 필터링 및 순위 조정
    filtered_candidates = filter_and_rank_drugs(candidates, patient_info, korean_symptom)
    
    #4단계: 최적 약품 후보 top-N 전달 (없으면 에러)
    if not filtered_candidates:
        return create_error_response("적합한 약품을 찾을 수 없습니다.", language)
    
    top_candidates = filtered_candidates[:5]  #LLM에 넘길 후보 수 조정 가능
    
    #5단계: LLM에 모든 결과를 한 번에 요청
    result = recommend_drug_llm_response(top_candidates, symptom, patient_info)
    
    return result

def filter_and_rank_drugs(candidates: List[Dict], patient_info: Dict, symptom: str) -> List[Dict]:
    """환자 정보를 고려하여 약품을 필터링하고 순위를 조정합니다."""
    if not patient_info:
        return candidates
    
    scored_candidates = []
    
    for drug in candidates:
        #기본 점수 (증상 매칭)
        score = 10
        warnings = []
        
        #GPT 기반 맥락 분석 (더 정교한 평가)
        context_score, context_warnings = analyze_drug_context(drug, patient_info, symptom)
        score += context_score
        warnings.extend(context_warnings)
        
        #기존 키워드 기반 분석 (백업)
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

def analyze_drug_context(drug: Dict, patient_info: Dict, symptom: str) -> tuple[int, List[str]]:
    """GPT를 활용한 맥락 기반 약품 분석"""
    try:
        openai.api_key = openai.api_key
        
        #약품 정보 구성
        drug_info = f"""
        약품명: {drug.get('itemName', '')}
        효능: {drug.get('efcyQesitm', '')}
        복용법: {drug.get('useMethodQesitm', '')}
        주의사항: {drug.get('atpnQesitm', '')}
        부작용: {drug.get('seQesitm', '')}
        상호작용: {drug.get('intrcQesitm', '')}
        """
        
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
        
        prompt = f"""당신은 의학 전문가입니다. 다음 환자 정보와 약품 정보를 분석하여 약품의 적합성을 평가해주세요.

환자 정보:
{patient_context}

약품 정보:
{drug_info}

다음 기준으로 평가해주세요:
1. 증상과 약품 효능의 적합성 (0-5점)
2. 환자 특성(나이, 성별)과 약품의 적합성 (0-3점)
3. 안전성 (알레르기, 상호작용, 부작용 고려) (0-5점)
4. 복용 편의성 (나이, 신체 조건 고려) (0-2점)

총점 (0-15점)과 위험 요소를 JSON 형식으로 응답해주세요:
{{
    "score": 점수,
    "risks": ["위험요소1", "위험요소2"],
    "reasoning": "평가 근거"
}}"""

        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        return result.get('score', 0), result.get('risks', [])
        
    except Exception as e:
        print(f"Error in context analysis: {e}")
        return 0, []

def analyze_drug_keywords(drug: Dict, patient_info: Dict) -> tuple[int, List[str]]:
    """키워드 기반 약품 분석 (환자 특성 고려)"""
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

def generate_pharmacist_questions(drug: Dict, symptom: str, language: str, patient_info: Dict) -> List[str]:
    """약사에게 할 질문을 생성합니다."""
    drug_name = drug.get('itemName', '')
    drug_purpose = drug.get('efcyQesitm', '')
    
    #환자 정보 요약 생성
    patient_summary = generate_patient_summary(patient_info, language)
    
    #증상을 한국어로 변환 (RAG 검색용)
    korean_symptom = symptom
    if language.lower() != "ko":
        korean_symptom = convert_symptom_to_korean(symptom, language)
    
    #질문 1: 약국에서 + 증상 관련 키워드 + 약 말씀해보세요
    question1 = f"약국에서 다음과 같이 말씀해보세요! : \"{korean_symptom}에 좋은 약을 찾고 있어요.\"\n약사님께 직접 말하시기 어려우시면, 아래 문장을 약사님께 보여주세요."
    
    #언어가 한국어가 아니면 부분 번역 (증상 부분 제외)
    if language.lower() != "ko":
        #증상 부분을 제외한 나머지 부분 번역
        prefix = "약국에서 다음과 같이 말씀해보세요! : "
        suffix = "약사님께 직접 말하시기 어려우시면, 아래 문장을 약사님께 보여주세요."
        
        translated_prefix = translate_text(prefix, language)
        translated_suffix = translate_text(suffix, language)
        
        question1 = f"{translated_prefix}\"{korean_symptom}에 좋은 약을 찾고 있어요.\"\n{translated_suffix}"
    
    #질문 2: 증상만 포함
    question2 = f'"{symptom}에 먹을 수 있는 약이 있을까요?"'
    
    #언어가 한국어가 아니면 번역 추가
    if language.lower() != "ko":
        translated_question2 = translate_text(question2, language)
        question2 = f"{question2}({translated_question2})"
    
    #질문 3: 환자 정보가 있을 때만 (증상 제외)
    question3 = ""
    if patient_summary:
        question3 = f'"{patient_summary}"'
        if language.lower() != "ko":
            translated_question3 = translate_text(question3, language)
            question3 = f"{question3}({translated_question3})"
    
    return [question1, question2, question3]

def generate_patient_summary(patient_info: Dict, language: str) -> str:
    """환자 정보를 요약하여 자연스러운 문장으로 생성합니다."""
    if not patient_info:
        return ""
    
    summary_parts = []
    
    #키/몸무게
    if patient_info.get('height') and patient_info.get('weight'):
        height = patient_info['height']
        weight = patient_info['weight']
        if language.lower() == "ko":
            summary_parts.append(f"키 {height}cm, 몸무게 {weight}kg")
        else:
            summary_parts.append(f"height {height}cm, weight {weight}kg")
    
    #알레르기
    if patient_info.get('allergies'):
        allergies = patient_info['allergies']
        if language.lower() == "ko":
            summary_parts.append(f"알레르기: {allergies}")
        else:
            #다른 언어로 입력된 경우 한국어 번역 추가
            translated_allergies = translate_text(allergies, "ko")
            summary_parts.append(f"알레르기: {translated_allergies}({allergies})")
    
    #현재 복용약
    if patient_info.get('current_medications'):
        medications = patient_info['current_medications']
        if language.lower() == "ko":
            summary_parts.append(f"현재 복용 중인 약: {medications}")
        else:
            #다른 언어로 입력된 경우 한국어 번역 추가
            translated_medications = translate_text(medications, "ko")
            summary_parts.append(f"현재 복용 중인 약: {translated_medications}({medications})")
    
    #과거 병력
    if patient_info.get('medical_history'):
        history = patient_info['medical_history']
        if language.lower() == "ko":
            summary_parts.append(f"과거 병력: {history}")
        else:
            #다른 언어로 입력된 경우 한국어 번역 추가
            translated_history = translate_text(history, "ko")
            summary_parts.append(f"과거 병력: {translated_history}({history})")
    
    return ", ".join(summary_parts)

def generate_warning_message(language: str) -> str:
    """경고 문구를 생성합니다."""
    if language.lower() == "ko":
        return "해당 내용은 정보 제공의 목적으로 구현되었으며, 약 복용에 대해서는 반드시 약사와 상담하세요."
    else:
        return translate_text("해당 내용은 정보 제공의 목적으로 구현되었으며, 약 복용에 대해서는 반드시 약사와 상담하세요.", language)

def create_error_response(message: str, language: str) -> Dict[str, Any]:
    """에러 응답을 생성합니다."""
    if language.lower() != "ko":
        message = translate_text(message, language)
    
    return {
        "drug_name": "",
        "drug_purpose": "",
        "drug_image_url": "",
        "pharmacist_question1": "",
        "pharmacist_question2": "",
        "pharmacist_question3": "",
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

def translate_drug_info(drugs: List[Dict], target_language: str) -> List[Dict]:
    """약품 정보를 대상 언어로 번역합니다."""
    for drug in drugs:
        #약품명 번역 및 로마자 표기
        if drug['itemName']:
            drug['itemName'] = translate_drug_name(drug['itemName'], target_language)
        
        #효능 번역
        if drug.get('efcyQesitm'):
            drug['efcyQesitm'] = translate_text(drug['efcyQesitm'], target_language)
        
        #주의사항 번역
        if drug.get('atpnQesitm'):
            drug['atpnQesitm'] = translate_text(drug['atpnQesitm'], target_language)
    
    return drugs

def translate_drug_name(drug_name: str, target_language: str) -> str:
    """약품명을 번역하고 로마자 표기를 추가합니다."""
    if target_language.lower() == "ko":
        return drug_name
    
    #기존 romanize_korean_names 함수 활용
    romanized_map = romanize_korean_names([drug_name])
    romanized_name = romanized_map.get(drug_name, drug_name)
    
    #번역
    translated_name = translate_text(drug_name, target_language)
    
    #원본(로마자) 형태로 반환
    return f"{drug_name}({romanized_name})"