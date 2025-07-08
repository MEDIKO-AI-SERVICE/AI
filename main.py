from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from hosp_utils.es_functions import query_elasticsearch_hosp, filtering_hosp
from pharm_utils.es_functions_for_pharmacy import query_elasticsearch_pharmacy
from hosp_utils.recommendation import HospitalRecommender
from pharm_utils.recommendation import PharmacyRecommender
from utils.feedback_manager import FeedbackManager

from er_utils.apis import *
from er_utils.direction_for_er import *
from er_utils.filtering_for_addr import *
from er_utils.for_redis import *

from utils.direction import calculate_travel_time_and_distance 
from utils.geocode import address_to_coords, coords_to_address
from utils.en_juso import get_english_address, get_korean_address
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

from gpt_utils.prompting_gpt import (
    analyze_symptoms,
    romanize_korean_names,
    select_department_for_symptoms,
    recommend_questions_to_doctor,
    get_body_part_adjectives,
    translate_text,
    summarize_symptom_keywords,
    summarize_symptom_input
)
from gpt_utils.department_mapping import DEPARTMENT_TRANSLATIONS

from medicine_rag_utils.drug_rag_manager import DrugRAGManager
import schedule
import threading
import mysql.connector
from datetime import datetime, timedelta
import configparser
from rag_utils.rag_search import search_similar_diseases
import pytz

from app_core import (
    get_db_connection,
    initialize_recommenders,
    retrain_models,
    run_scheduler,
    update_drug_data,
    hospital_recommender,
    pharmacy_recommender,
    drug_rag_manager
)

app = FastAPI()

#스케줄러 시작
@app.on_event("startup")
async def startup_event():
    print("Starting up the application...")
    
    # 스케줄러 시작
    threading.Thread(target=run_scheduler, daemon=True).start()
    
    # 추천 모델 초기화 (비동기로)
    def init_recommenders():
        try:
            initialize_recommenders()
            print("Recommenders initialized successfully")
        except Exception as e:
            print(f"Error initializing recommenders: {e}")
    
    threading.Thread(target=init_recommenders, daemon=True).start()
    
    # 약품 RAG 시스템 초기화 (완전히 백그라운드로)
    def init_drug_rag():
        try:
            print("Initializing drug RAG system in background...")
            global drug_rag_manager
            drug_rag_manager = DrugRAGManager()
            success = drug_rag_manager.initialize_rag_system()
            if success:
                print("Drug RAG system initialized successfully")
            else:
                print("Failed to initialize drug RAG system")
        except Exception as e:
            print(f"Error initializing drug RAG system: {e}")
    
    # 백그라운드 스레드에서 RAG 시스템 초기화
    threading.Thread(target=init_drug_rag, daemon=True).start()
    
    print("Application startup completed - services will be available as they initialize")
    
@app.post("/api/recommend/hospital")
async def recommend_hospital(request: Request):
    global hospital_recommender
    if not hospital_recommender:
        initialize_recommenders()

    data = await request.json()
    print(f"[hospital] Request data: {data}", flush=True)

    basic_info = data.get("basic_info")
    health_info = data.get("health_info")
    department = data.get("department", "내과")
    # department가 리스트인 경우 문자열로 변환, 문자열인 경우 그대로 사용
    if isinstance(department, list):
        department = ", ".join(department)
    suspected_disease = data.get("suspected_disease")
    primary_hospital = data.get("primary_hospital", False)
    secondary_hospital = data.get("secondary_hospital", False)
    tertiary_hospital = data.get("tertiary_hospital", False)
    member_id = data.get("member_id")
    user_lat = data.get("lat")
    user_lon = data.get("lon")
    sort_type = data.get("sort_type", "distance")  # "recommend" 또는 "distance"

    try:
        coords = address_to_coords(basic_info['address'])
        if "error" in coords:
            return JSONResponse(content={"error": coords["error"]}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    if not user_lat or not user_lon:
        user_lat = coords['lat']
        user_lon = coords['lon']

    es_results = query_elasticsearch_hosp(user_lat, user_lon, department, primary_hospital, secondary_hospital, tertiary_hospital)
    if "hits" not in es_results or not es_results["hits"]["hits"]:
        return JSONResponse(content={"message": "No hospitals found"}, status_code=404)

    filtered_hospitals = filtering_hosp(es_results)
    df = pd.DataFrame([hospital for hospital in filtered_hospitals])

    with ThreadPoolExecutor(max_workers=10) as executor:
        travel_infos = list(
            executor.map(
                lambda row: calculate_travel_time_and_distance(row, user_lat, user_lon),
                df.to_dict("records")
            )
        )

    df["travel_info"] = travel_infos
    df["transit_travel_distance_km"] = df["travel_info"].apply(lambda x: x.get("transit_travel_distance_km") if x else None)
    df["transit_travel_time_h"] = df["travel_info"].apply(lambda x: x.get("transit_travel_time_h") if x else None)
    df["transit_travel_time_m"] = df["travel_info"].apply(lambda x: x.get("transit_travel_time_m") if x else None)
    df["transit_travel_time_s"] = df["travel_info"].apply(lambda x: x.get("transit_travel_time_s") if x else None)
    df.drop(columns=["travel_info"], inplace=True)

    # 정렬 타입에 따른 처리
    if sort_type == "distance":
        # 거리순 정렬 (강화학습 없이 거리만 고려)
        total_travel_time = (
            df['transit_travel_time_h'] * 3600 +
            df['transit_travel_time_m'] * 60 +
            df['transit_travel_time_s']
        )
        df['similarity'] = 1 / (1 + total_travel_time)  # 거리가 가까울수록 높은 점수
        recommended_hospitals = df.sort_values(by=["similarity"], ascending=[False])
    else:
        # 추천순 정렬 (기존 강화학습 로직)
        user_embedding = hospital_recommender.embed_user_profile(basic_info, health_info, suspected_disease, department)
        hospital_embeddings = hospital_recommender.embed_hospital_data(df)

        recommended_hospitals = hospital_recommender.recommend_hospitals(
            user_embedding=user_embedding,
            hospital_embeddings=hospital_embeddings,
            hospitals_df=df,
            member_id=member_id
        )

    for col in ["transit_travel_time_h", "transit_travel_time_m", "transit_travel_time_s"]:
        recommended_hospitals.loc[recommended_hospitals[col].isnull(), col] = 0

    recommended_hospitals.reset_index(drop=True, inplace=True)

    if basic_info.get("language", "").lower() != "ko":
        with ThreadPoolExecutor(max_workers=10) as executor:
            recommended_hospitals["eng_address"] = recommended_hospitals["address"].apply(get_english_address)
        recommended_hospitals = recommended_hospitals[recommended_hospitals["eng_address"].notnull()]
        recommended_hospitals["address"] = recommended_hospitals["eng_address"]
        recommended_hospitals.drop(columns=["eng_address"], inplace=True)

    names = recommended_hospitals["name"].tolist()
    romanized_map = romanize_korean_names(names)
    recommended_hospitals["name"] = recommended_hospitals["name"].map(
        lambda x: f"{x} ({romanized_map.get(x)})" if romanized_map.get(x) else x
    )

    return JSONResponse(content=recommended_hospitals.to_dict(orient="records"))

@app.post("/api/recommend/pharmacy")
async def recommend_pharmacy(request: Request):
    global pharmacy_recommender
    if not pharmacy_recommender:
        initialize_recommenders()
    
    data = await request.json()
    print(f"[pharmacy] Request data: {data}", flush=True)

    user_lat = data.get('lat')
    user_lon = data.get('lon')
    basic_info = data.get("basic_info")
    member_id = data.get("member_id")
    sort_type = data.get("sort_type", "distance")  # "recommend" 또는 "distance"

    try:
        coords = address_to_coords(basic_info['address'])
        if "error" in coords:
            return JSONResponse(content={"error": coords["error"]}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    if not user_lat or not user_lon:
        user_lat = coords['lat']
        user_lon = coords['lon']

    es_results = query_elasticsearch_pharmacy(user_lat, user_lon)

    if "hits" in es_results and es_results['hits']['total']['value'] > 0:
        pharmacy_data = [hit['_source'] for hit in es_results['hits']['hits']]
        df = pd.DataFrame(pharmacy_data)

        df.rename(columns={
            'wgs84lat': 'latitude',
            'wgs84lon': 'longitude',
            'dutyaddr': 'address'
        }, inplace=True)

        with ThreadPoolExecutor(max_workers=10) as executor:
            travel_infos = list(executor.map(
                lambda row: calculate_travel_time_and_distance(row, user_lat, user_lon),
                df.to_dict("records")
            ))

        df['travel_info'] = travel_infos

        df["transit_travel_distance_km"] = df['travel_info'].apply(lambda x: x.get("transit_travel_distance_km") if x else None)
        df["transit_travel_time_h"] = df['travel_info'].apply(lambda x: x.get("transit_travel_time_h") if x else None)
        df["transit_travel_time_m"] = df['travel_info'].apply(lambda x: x.get("transit_travel_time_m") if x else None)
        df["transit_travel_time_s"] = df['travel_info'].apply(lambda x: x.get("transit_travel_time_s") if x else None)

        df.drop(columns=["travel_info"], inplace=True)

        for col in ["transit_travel_distance_km", "transit_travel_time_h", "transit_travel_time_m", "transit_travel_time_s"]:
            df.loc[df[col].isnull(), col] = 0

        # 정렬 타입에 따른 처리
        if sort_type == "distance":
            # 거리순 정렬 (강화학습 없이 거리만 고려)
            total_travel_time = (
                df['transit_travel_time_h'] * 3600 +
                df['transit_travel_time_m'] * 60 +
                df['transit_travel_time_s']
            )
            df['similarity'] = 1 / (1 + total_travel_time)  # 거리가 가까울수록 높은 점수
            recommended_pharmacies = df.sort_values(by=["similarity"], ascending=[False])
        else:
            # 추천순 정렬 (기존 강화학습 로직)
            recommended_pharmacies = pharmacy_recommender.recommend_pharmacies(df, member_id=member_id)

        recommended_pharmacies = recommended_pharmacies.sort_values(by=["similarity"], ascending=[False])
        recommended_pharmacies = recommended_pharmacies.reset_index(drop=True)

        if basic_info.get("language").lower() != "ko":
            with ThreadPoolExecutor(max_workers=10) as executor:
                recommended_pharmacies["eng_address"] = recommended_pharmacies["address"].apply(get_english_address)
            recommended_pharmacies = recommended_pharmacies[recommended_pharmacies["eng_address"].notnull()]
            recommended_pharmacies["address"] = recommended_pharmacies["eng_address"]
            recommended_pharmacies.drop(columns=["eng_address"], inplace=True)

        names = recommended_pharmacies["dutyname"].tolist()
        romanized_map = romanize_korean_names(names)
        recommended_pharmacies["dutyname"] = recommended_pharmacies["dutyname"].map(
            lambda x: f"{x} ({romanized_map.get(x)})" if romanized_map.get(x) else x
        )

        return JSONResponse(content=recommended_pharmacies.to_dict(orient="records"))

    else:
        return JSONResponse(content={"message": "No pharmacies found"}, status_code=404)

@app.post("/api/recommend/er")
async def recommend_er(request: Request):
    data = await request.json()
    print(f"ER Request data: {data}", flush=True)
    conditions_korean = data.get('conditions', [])

    address_filter = AddressFilter()

    user_lat = data.get('lat')
    user_lon = data.get('lon')
    basic_info = data.get('basic_info', {})
    address = basic_info['address']

    try:
        coords = address_to_coords(address)
        if "error" in coords:
            return JSONResponse(content={"error": coords["error"]}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    if not user_lat or not user_lon:
        user_lat = coords['lat']
        user_lon = coords['lon']
    else:
        try:
            converted_address = coords_to_address(user_lat, user_lon)
            if "error" not in converted_address:
                converted_coords = address_to_coords(converted_address['address_name'])

                if converted_coords['lat'] != coords['lat'] or converted_coords['lon'] != coords['lon']:
                    lat_diff = abs(converted_coords['lat'] - coords['lat'])
                    lon_diff = abs(converted_coords['lon'] - coords['lon'])

                    if lat_diff < 0.00001 and lon_diff < 0.00001:
                        pass
                    else:
                        address = converted_address['address_name']
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    stage1, stage2 = address.split()[:2]

    condition_mapping = {
        "조산산모": "MKioskTy8",
        "정신질환자": "MKioskTy9",
        "신생아": "MKioskTy10",
        "중증화상": "MKioskTy11"
    }
    conditions = [condition_mapping[cond] for cond in conditions_korean if cond in condition_mapping]

    hpid_list = get_hospitals_by_condition(stage1, stage2, conditions)
    if not hpid_list:
        return JSONResponse(content={"message": "No hospitals found for the given conditions"}, status_code=404)

    real_time_data = get_real_time_bed_info(stage1, stage2, hpid_list)
    if not real_time_data:
        return JSONResponse(content={"message": "No real-time bed information available"}, status_code=404)

    df = pd.DataFrame(real_time_data)
    enriched_df = address_filter.enrich_filtered_df(df)
    enriched_df = calculate_travel_time_and_sort(enriched_df, user_lat, user_lon)

    columns_to_return = [
        "dutyName", "dutyAddr", "dutyTel3", "hvamyn", "is_trauma",
        "transit_travel_distance_km", "transit_travel_time_h",
        "transit_travel_time_m", "transit_travel_time_s", "wgs84Lat", "wgs84Lon"
    ]
    filtered_df = enriched_df[columns_to_return].copy()

    for col in ["transit_travel_distance_km", "transit_travel_time_h", "transit_travel_time_m", "transit_travel_time_s"]:
        filtered_df.loc[filtered_df[col].isnull(), col] = 0

    if basic_info.get("language").lower() != "ko":
        filtered_df["eng_address"] = filtered_df["dutyAddr"].apply(get_english_address)
        filtered_df = filtered_df[filtered_df["eng_address"].notnull()]
        filtered_df["dutyAddr"] = filtered_df["eng_address"]
        filtered_df.drop(columns=["eng_address"], inplace=True)

    names = filtered_df["dutyName"].tolist()
    romanized_map = romanize_korean_names(names)
    filtered_df["dutyName"] = filtered_df["dutyName"].map(
        lambda x: f"{x} ({romanized_map.get(x)})" if romanized_map.get(x) else x
    )

    return JSONResponse(content=filtered_df.to_dict(orient='records'))


#신체 부위 -> 형용사 3개
@app.post("/api/pre_question/expression")
async def get_expression_adjectives(request: Request):
    try:
        data = await request.json()
        print(f"Expression Request data: {data}", flush=True)

        body_part = data.get('body_part', '')
        language = data.get('language', '').upper()

        if not body_part or not language:
            return JSONResponse(content={"error": "Both 'body_part' and 'language' are required"}, status_code=400)

        # GPT를 통한 형용사 생성
        adjectives = get_body_part_adjectives(body_part, language)

        final_response = {
            "adjectives": adjectives
        }

        print("final_response:", final_response, flush=True)
        return JSONResponse(content=final_response, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



import math

def clean_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj

#증상 -> 약품 추천 및 약사 질문
@app.post("/api/pre_question/1")
async def recommend_drug_for_symptom(request: Request):
    try:
        data = await request.json()
        print(f"Drug recommendation request data: {data}", flush=True)

        symptom = data.get('sign', '')
        language = data.get('language', 'ko').lower()
        patient_info = data.get('patient_info', {})

        if not symptom:
            return JSONResponse(content={"error": "Symptom is required"}, status_code=400)

        # RAG 시스템이 준비되지 않은 경우
        if not drug_rag_manager or not drug_rag_manager.is_ready():
            if drug_rag_manager and drug_rag_manager.is_initializing:
                return JSONResponse(
                    content={
                        "error": "Drug recommendation system is still initializing. Please try again in a few minutes.",
                        "status": "initializing"
                    }, 
                    status_code=503
                )
            else:
                return JSONResponse(
                    content={
                        "error": "Drug recommendation system is not available.",
                        "status": "unavailable"
                    }, 
                    status_code=503
                )

        # RAG 시스템을 통한 약품 추천 (환자 정보 포함)
        result = drug_rag_manager.get_recommendation(symptom, language, patient_info)
        
        if not result:
            return JSONResponse(content={"error": "Failed to get drug recommendation"}, status_code=500)

        final_response = {
            "drug_name": result["drug_name"],
            "drug_purpose": result["drug_purpose"],
            "drug_image_url": result["drug_image_url"],
            "pharmacist_question1": result["pharmacist_questions"][0] if len(result["pharmacist_questions"]) > 0 else "",
            "pharmacist_question2": result["pharmacist_questions"][1] if len(result["pharmacist_questions"]) > 1 else "",
            "pharmacist_question3": result["pharmacist_questions"][2] if len(result["pharmacist_questions"]) > 2 else ""
        }

        print("Drug recommendation response:", final_response, flush=True)
        final_response = clean_nan(final_response)
        return JSONResponse(content=final_response, status_code=200)

    except Exception as e:
        print(f"Error in drug recommendation: {e}", flush=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

#RAG 시스템 상태 확인
@app.get("/api/drug-rag/status")
async def get_drug_rag_status():
    try:
        global drug_rag_manager
        if not drug_rag_manager:
            return JSONResponse(content={"error": "Drug RAG system not initialized"}, status_code=500)
        
        status = drug_rag_manager.get_system_status()
        
        # 추가 정보
        status.update({
            "system_ready": drug_rag_manager.is_ready()
        })
        
        return JSONResponse(content=status, status_code=200)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#진료과 설명 딕셔너리
DEPARTMENT_DESCRIPTIONS = {
    "가정의학과": "이 진료과는 전 연령대의 환자를 대상으로 예방, 진단, 치료, 건강관리를 담당합니다.",
    "내과": "이 진료과는 내분비, 소화기, 호흡기, 심혈관 등 내부 장기에서 발생한 질환을 비수술적 방식으로 치료합니다.",
    "정형외과": "이 진료과는 뼈, 관절, 근육 등 근골격계 질환을 진단하고 치료합니다.",
    "한방과":"이 진료과는 한국 전통 치료법을 중심으로 질환을 처치 및 예방합니다.",
    "피부과":"이 진료과는 피부와 피부에 부속된 기관의 질병을 치료합니다.",
    "치의과":"이 진료과는 치아와 치아에 부석된 기관(치주조직, 구강구조물, 턱뼈, 턱관절, 얼굴)의 질환을 치료합니다.",
    "정신건강의학과":"이 진료과는 면담과 검사를 통해 정신적 질병(조현병, 양극성 장애, 우울 장애, 강박 장애 등)을 치료합니다.",
    "재활의학과":"이 진료과는 각종 질병 및 사고로 인하여 장애가 생긴 환자의 재활을 목적으로 합니다.",
    "이비인후과":"이 진료과는 귀, 코와 관련된 질환 및 두경부 외과 질환을 치료합니다.",
    "외과":"이 진료과는 몸 외부의 상처나 내장 기관의 질병을 수술이나 그와 비슷한 방법으로 치료합니다.",
    "예방의학과":"이 진료과는 인구집단이 가진 보건문제 예방과 건강 증진을 목적으로 합니다.",
    "영상의학과":"이 진료과는 다양한 영상장비를 이용하여(X-선검사, 초음파검사, CT검사, MRI검사, 골밀도검사, 유방촬영) 영상을 획득하고 영상 자료를 토대로 질병을 진단합니다.",
    "안과":"이 진료과는 눈과 관련된 질환을 진단 및 치료합니다.",
    "심장혈관흉부외과":"이 진료과는 흉부에 위치한 심장, 폐, 식도, 대동맥, 종격동, 횡격막, 기관 등의 질환에 대한 수술적으로 치료합니다.",
    "신경외과":"이 진료과는 뇌와 척수를 포함한 중추신경계와 말초신경계에서 발생하는 질병을 진단하고 수술적인 방법으로 치료합니다.",
    "신경과":"이 진료과는 뇌와 척수를 포함한 중추신경계와 말초신경계에서 발생하는 질병을 진단하고 비수술적인 방법으로 치료합니다.",
    "소아청소년과":"이 진료과는 영아기부터 청소년기까지의 환자를 중심으로 질환을 진료 및 치료합니다.",
    "성형외과":"이 진료과는 재건 수술과 미용 수술을 중심으로 인체 외형 변화를 제공합니다.",
    "산부인과":"이 진료과는 모든 연령의 여성을 대상으로 임신, 출산, 여성 생식 기관 질병을 진료 및 치료합니다.",
    "비뇨의학과":"이 진료과는 콩팥(신장), 요관, 방광, 요도 등 요로계 장기들과 음경, 고환, 정관 및 전립선 등 남성생식과 관련된 장기 관련 질환을 진단하고 주로 수술적인 방법으로 치료합니다.",
    "마취통증의학과":"이 진료과는 각종 수술과 진정 관리를 받는 환자의 안전을 책임지며, 통증 클리닉을 통해 통증 환자들에 대한 진료를 담당합니다."
}


@app.post("/api/pre_question/2")
async def pre_question_2(request: Request):
    try:
        data = await request.json()
        language = data.get('language', 'KO').upper()
        bodypart = data.get('bodypart', '')
        selectedSign = data.get('selectedSign', '')
        symptom_info = data.get('symptom', {})
        intensity = symptom_info.get('intensity', '')
        startDate = symptom_info.get('startDate', '')
        additional = symptom_info.get('additional', '')

        symptoms_for_llm = {
            "bodypart": bodypart,
            "selectedSign": selectedSign,
            "intensity": intensity,
            "startDate": startDate,
            "additional": additional
        }

        # 1. 진료과 선정
        department_ko = select_department_for_symptoms(symptoms_for_llm)
        # 2. 진료과 설명
        department_description = DEPARTMENT_DESCRIPTIONS.get(department_ko, "")
        # 3. 진료과 필드 (언어별)
        if language == "KO":
            department_field = department_ko
        else:
            romanized = romanize_korean_names([department_ko]).get(department_ko, "")
            # 딕셔너리에서 직접 영어 번역 가져오기
            english_translation = DEPARTMENT_TRANSLATIONS.get(department_ko, {}).get("EN", "")
            department_field = f"{department_ko} ({romanized}, {english_translation})"
            department_description = translate_text(department_description, target_language=language.lower())
        # 4. 의사 질문 추천
        questions_ko = recommend_questions_to_doctor(symptoms_for_llm)
        questions_to_doctor = []
        for q_ko in questions_ko:
            if language == "KO":
                questions_to_doctor.append(q_ko)
            else:
                q_trans = translate_text(q_ko, target_language=language.lower())
                questions_to_doctor.append(f"{q_ko} ({q_trans})")
        return JSONResponse(content={
            "department": department_field,
            "department_description": department_description,
            "questions_to_doctor": questions_to_doctor,
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)




#증상, 언어 -> 병명, 질문&체크리스트
@app.post("/api/pre_question/3")
async def process_symptoms(request: Request):
    try:
        data = await request.json()
        print(f"Request data: {data}", flush=True)

        # 1. 파라미터 파싱
        language = data.get('language', 'KO').upper()
        patientinfo = data.get('patientinfo', {})
        bodypart = data.get('bodypart', '')
        selectedSign = data.get('selectedSign', '')
        symptom_info = data.get('symptom', {})
        intensity = symptom_info.get('intensity', '')
        startDate = symptom_info.get('startDate', '')
        durationValue = symptom_info.get('durationValue', '')
        durationUnit = symptom_info.get('durationUnit', '')
        state = symptom_info.get('state', '')
        additional = symptom_info.get('additional', '')

        # analyze_symptoms에 넘길 증상 정보 (영어)
        symptoms_for_llm = {
            "bodypart": bodypart,
            "selectedSign": selectedSign,
            "intensity": intensity,
            "startDate": startDate,
            "duration": f"{durationValue} {durationUnit}",
            "state": state,
            "additional": additional
        }

        # 2. LLM 분석
        analysis = analyze_symptoms(symptoms_for_llm, patientinfo, exclude_possible_conditions=True)

        # 3. 한국 기준 날짜/시간
        now_kst = datetime.utcnow() + timedelta(hours=9)
        now_kst_str = now_kst.strftime("%Y-%m-%d %H:%M:%S")

        # 4. 증상 관련 입력 요약(최대 300자, 사용자 언어)
        symptom_input_text = f"신체 부위: {bodypart}\n증상 설명: {selectedSign}\n강도: {intensity}\n시작일: {startDate}\n지속 기간: {durationValue} {durationUnit}\n상태: {state}\n추가 설명: {additional}"
        symptom_summary_ko = summarize_symptom_input(symptom_input_text, language="KO")
        if language != "KO":
            symptom_summary_trans = summarize_symptom_input(symptom_input_text, language=language)
            symptom_summary = f"{symptom_summary_ko} ({symptom_summary_trans})"
        else:
            symptom_summary = symptom_summary_ko

        # 5. 증상 키워드 기반 50자 이내 한 줄 요약 생성
        summary_keywords = []
        if bodypart:
            summary_keywords.append(str(bodypart))
        if selectedSign:
            summary_keywords.append(str(selectedSign))
        if durationValue or durationUnit:
            summary_keywords.append(f"{durationValue}{durationUnit}".strip())
        if state:
            summary_keywords.append(str(state))
        if additional:
            summary_keywords.append(str(additional))
        summary_text = summarize_symptom_keywords(summary_keywords, language)

        # 6. 진료과 번역: 한국어가 아니면 '한국어(자국어)' 형태
        department_ko = analysis["department_ko"]
        department_description = DEPARTMENT_DESCRIPTIONS.get(department_ko, "")
        if language == "KO":
            department_field = department_ko
        else:
            romanized = romanize_korean_names([department_ko]).get(department_ko, "")
            english_translation = DEPARTMENT_TRANSLATIONS.get(department_ko, {}).get("EN", "")
            department_field = f"{department_ko} ({romanized}, {english_translation})"
        
        # 7. 의사에게 할 질문 추천
        questions_to_doctor_list = recommend_questions_to_doctor(symptoms_for_llm)
        questions_to_doctor_trans = {}
        for idx, question_ko in enumerate(questions_to_doctor_list):
            if language == "KO":
                questions_to_doctor_trans[f"question {idx+1}"] = question_ko
            else:
                q_trans = translate_text(question_ko, target_language=language.lower())
                questions_to_doctor_trans[f"question {idx+1}"] = f"{question_ko} ({q_trans})"

        final_response = {
            "created_at_kst": now_kst_str,
            "summary": summary_text,
            "department": department_field,
            "department_description": department_description,
            "symptom_summary": symptom_summary,
            "questions_to_doctor": questions_to_doctor_trans
        }
        return JSONResponse(content=final_response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)