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
    analyze_symptoms_with_summary,
    romanize_korean_names,
    select_department_for_symptoms,
    recommend_questions_to_doctor,
    get_body_part_adjectives,
    translate_text,
    recommend_drug_llm_response
)
from gpt_utils.department_mapping import DEPARTMENT_TRANSLATIONS, DEPARTMENT_ROMANIZATION, DEPARTMENT_DESCRIPTIONS

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

    # 언어가 한국어가 아닐 때만 로마자 표기 추가
    if basic_info.get("language", "").lower() != "ko":
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

        # 언어가 한국어가 아닐 때만 로마자 표기 추가
        if basic_info.get("language").lower() != "ko":
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

    # 언어가 한국어가 아닐 때만 로마자 표기 추가
    if basic_info.get("language").lower() != "ko":
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
    start_total = time.time()
    try:
        data = await request.json()
        print(f"Drug recommendation request data: {data}", flush=True)
        t0 = time.time()
        symptom = data.get('sign', '')
        language = data.get('language', 'ko').lower()
        patient_info = data.get('patient_info', {})
        print(f"[latency] 파라미터 파싱: {time.time() - t0:.3f}초", flush=True)
        t0 = time.time()
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
        print(f"[latency] RAG 시스템 준비 확인: {time.time() - t0:.3f}초", flush=True)
        t0 = time.time()
        result = drug_rag_manager.get_recommendation(symptom, language, patient_info)
        print(f"[latency] RAG 추천 및 LLM 호출: {time.time() - t0:.3f}초", flush=True)
        t0 = time.time()
        if not result:
            return JSONResponse(content={"error": "Failed to get drug recommendation"}, status_code=500)
        # Translate only drug_purpose if language is not Korean
        drug_purpose = result.get("drug_purpose", "")
        if language != "ko" and language != "KO":
            try:
                drug_purpose = translate_text(drug_purpose, target_language=language)
            except Exception as e:
                print(f"Error translating drug_purpose: {e}")
        final_response = {
            "drug_name": result["drug_name"],
            "drug_purpose": drug_purpose,
            "drug_image_url": result["drug_image_url"],
            "pharmacist_question1": result["pharmacist_questions"][0] if len(result["pharmacist_questions"]) > 0 else "",
            "pharmacist_question2": result["pharmacist_questions"][1] if len(result["pharmacist_questions"]) > 1 else "",
            "pharmacist_question3": result["pharmacist_questions"][2] if len(result["pharmacist_questions"]) > 2 else ""
        }
        print(f"[latency] 응답 데이터 가공: {time.time() - t0:.3f}초", flush=True)
        print(f"[latency] 전체 소요 시간: {time.time() - start_total:.3f}초", flush=True)
        final_response = clean_nan(final_response)
        return JSONResponse(content=final_response, status_code=200)
    except Exception as e:
        print(f"Error in drug recommendation: {e}", flush=True)
        print(f"[latency] 전체 소요 시간(예외): {time.time() - start_total:.3f}초", flush=True)
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




@app.post("/api/pre_question/2")
async def pre_question_2(request: Request):
    start_total = time.time()
    try:
        data = await request.json()
        t0 = time.time()
        language = data.get('language', 'KO').upper()
        bodypart = data.get('bodypart', '')
        selectedSign = data.get('selectedSign', '')
        symptom_info = data.get('symptom', {})
        intensity = symptom_info.get('intensity', '')
        startDate = symptom_info.get('startDate', '')
        additional = symptom_info.get('additional', '')
        print(f"[latency] 파라미터 파싱: {time.time() - t0:.3f}초", flush=True)
        t0 = time.time()
        symptoms_for_llm = {
            "bodypart": bodypart,
            "selectedSign": selectedSign,
            "intensity": intensity,
            "startDate": startDate,
            "additional": additional
        }
        department_ko = select_department_for_symptoms(symptoms_for_llm)
        print(f"[latency] 진료과 선정(LLM): {time.time() - t0:.3f}초", flush=True)
        t0 = time.time()
        # 진료과 설명 (언어별)
        department_description_dict = DEPARTMENT_DESCRIPTIONS.get(department_ko, {})
        department_description = department_description_dict.get(language, department_description_dict.get("KO", ""))
        
        translations = DEPARTMENT_TRANSLATIONS.get(department_ko, {})
        romanized = DEPARTMENT_ROMANIZATION.get(department_ko, "")
        translated = translations.get(language, department_ko)  # fallback to Korean if not found
        if language == "KO":
            department_field = department_ko
        else:
            department_field = f"{department_ko} ({romanized}, {translated})"
        print(f"[latency] 진료과 설명/번역: {time.time() - t0:.3f}초", flush=True)
        t0 = time.time()
        questions_ko = recommend_questions_to_doctor(symptoms_for_llm)
        questions_to_doctor = []
        for q_ko in questions_ko:
            if language == "KO":
                questions_to_doctor.append(q_ko)
            else:
                q_trans = translate_text(q_ko, target_language=language.lower())
                questions_to_doctor.append(f"{q_ko} ({q_trans})")
        print(f"[latency] 의사 질문 추천/번역: {time.time() - t0:.3f}초", flush=True)
        print(f"[latency] 전체 소요 시간: {time.time() - start_total:.3f}초", flush=True)
        return JSONResponse(content={
            "department": department_field,
            "department_description": department_description,
            "questions_to_doctor": questions_to_doctor,
        })
    except Exception as e:
        print(f"[latency] 전체 소요 시간(예외): {time.time() - start_total:.3f}초", flush=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)




#증상, 언어 -> 병명, 질문&체크리스트
@app.post("/api/pre_question/3")
async def process_symptoms(request: Request):
    start_total = time.time()
    try:
        data = await request.json()
        print(f"Request data: {data}", flush=True)
        t0 = time.time()
        language = data.get('language', 'KO').upper()
        patientinfo = data.get('patientinfo', {})
        gender = patientinfo.get('gender', '')
        age_str = str(patientinfo.get('age', ''))
        allergy = patientinfo.get('allergy', '')
        nowMedicine = patientinfo.get('nowMedicine', '')
        pastHistory = patientinfo.get('pastHistory', '')
        familyHistory = patientinfo.get('familyHistory', '')
        bodypart = data.get('bodypart', '') or data.get('body_part', '')
        selectedSign = data.get('selectedSign', '')
        if isinstance(selectedSign, list):
            selectedSign = ', '.join(str(s) for s in selectedSign)
        symptom_info = data.get('symptom', {})
        intensity = symptom_info.get('intensity', '')
        startDate = symptom_info.get('startDate', '')
        durationValue = symptom_info.get('durationValue', '')
        durationUnit = symptom_info.get('durationUnit', '')
        state = symptom_info.get('state', '')
        additional = symptom_info.get('additional', '')
        print(f"[DEBUG] bodypart: {bodypart}")
        print(f"[DEBUG] selectedSign: {selectedSign}")
        print(f"[DEBUG] intensity: {intensity}")
        print(f"[DEBUG] startDate: {startDate}")
        print(f"[DEBUG] durationValue: {durationValue}")
        print(f"[DEBUG] durationUnit: {durationUnit}")
        print(f"[DEBUG] state: {state}")
        print(f"[DEBUG] additional: {additional}")
        print(f"[DEBUG] patientinfo: {patientinfo}")
        print(f"[latency] 파라미터 파싱: {time.time() - t0:.3f}초", flush=True)
        t0 = time.time()
        symptoms_for_llm = {
            "bodypart": str(bodypart) if bodypart is not None else '',
            "selectedSign": str(selectedSign) if selectedSign is not None else '',
            "intensity": str(intensity) if intensity is not None else '',
            "startDate": str(startDate) if startDate is not None else '',
            "duration": f"{durationValue} {durationUnit}",
            "state": str(state) if state is not None else '',
            "additional": str(additional) if additional is not None else '',
            "gender": str(gender) if gender is not None else '',
            "age": str(age_str) if age_str is not None else '',
            "allergy": str(allergy) if allergy is not None else '',
            "nowMedicine": str(nowMedicine) if nowMedicine is not None else '',
            "pastHistory": str(pastHistory) if pastHistory is not None else '',
            "familyHistory": str(familyHistory) if familyHistory is not None else ''
        }
        analysis_result = analyze_symptoms_with_summary(symptoms_for_llm, patientinfo, language)
        print(f"[latency] 증상 분석(LLM): {time.time() - t0:.3f}초", flush=True)
        t0 = time.time()
        now_kst = datetime.utcnow() + timedelta(hours=9)
        now_kst_str = now_kst.strftime("%Y-%m-%d %H:%M:%S")
        department_ko = analysis_result["department_ko"]
        symptom_summary = analysis_result["symptom_summary"]
        summary_text = analysis_result["summary_text"]
        # 진료과 설명 (언어별)
        department_description_dict = DEPARTMENT_DESCRIPTIONS.get(department_ko, {})
        department_description = department_description_dict.get(language, department_description_dict.get("KO", ""))
        
        translations = DEPARTMENT_TRANSLATIONS.get(department_ko, {})
        romanized = DEPARTMENT_ROMANIZATION.get(department_ko, "")
        translated = translations.get(language, department_ko)  # fallback to Korean if not found
        if language == "KO":
            department_field = department_ko
        else:
            department_field = f"{department_ko} ({romanized}, {translated})"
        questions_to_doctor_list = recommend_questions_to_doctor(symptoms_for_llm)
        questions_to_doctor_trans = {}
        for idx, question_ko in enumerate(questions_to_doctor_list):
            if language == "KO":
                questions_to_doctor_trans[f"question {idx+1}"] = question_ko
            else:
                q_trans = translate_text(question_ko, target_language=language.lower())
                questions_to_doctor_trans[f"question {idx+1}"] = f"{question_ko} ({q_trans})"
        print(f"[latency] 진료과/질문 추천/번역(LLM): {time.time() - t0:.3f}초", flush=True)
        print(f"[latency] 전체 소요 시간: {time.time() - start_total:.3f}초", flush=True)
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
        print(f"[latency] 전체 소요 시간(예외): {time.time() - start_total:.3f}초", flush=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)