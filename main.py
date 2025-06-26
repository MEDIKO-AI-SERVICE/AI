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

from gpt_utils.prompting_gpt import *
import schedule
import threading
import mysql.connector
from datetime import datetime
import configparser

app = FastAPI()


#DB 연결 설정
def get_db_connection():
    config=configparser.ConfigParser()
    config.read('keys.config')
    
    return mysql.connector.connect(
        host=config['DB_INFO']['host'],
        user=config['DB_INFO']['id'],
        password=config['DB_INFO']['password'],
        database=config['DB_INFO']['db']
    )

#추천 모델 인스턴스
hospital_recommender=None
pharmacy_recommender=None

def initialize_recommenders():
    #추천 모델 초기화
    global hospital_recommender, pharmacy_recommender
    db_connection=get_db_connection()
    hospital_recommender=HospitalRecommender(
        db_connection=db_connection
    )
    pharmacy_recommender=PharmacyRecommender(
        db_connection=db_connection
    )

def retrain_models():
    #모델 재훈련
    global hospital_recommender, pharmacy_recommender
    
    #최근 7일간 선택한 사용자들의 member_id 가져오기
    query="""
    SELECT DISTINCT member_id 
    FROM (
        SELECT member_id, selected_at FROM selected_hp
        UNION
        SELECT member_id, selected_at FROM selected_ph
    ) combined
    WHERE selected_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
    """
    
    try:
        db_connection=get_db_connection()
        member_ids=pd.read_sql(query, db_connection)['member_id'].tolist()
        
        for member_id in member_ids:
            if hospital_recommender:
                hospital_recommender.update_from_feedback(member_id)
            if pharmacy_recommender:
                pharmacy_recommender.update_from_feedback(member_id)
                
    except Exception as e:
        print(f"모델 재훈련 중 오류 발생: {e}")
    finally:
        if 'db_connection' in locals():
            db_connection.close()

def run_scheduler():
    #스케줄러 실행
    schedule.every().day.at("03:00").do(retrain_models)  #매일 새벽 3시에 재훈련
    
    while True:
        schedule.run_pending()
        time.sleep(60)

#스케줄러 시작
@app.on_event("startup")
def startup_event():
    threading.Thread(target=run_scheduler, daemon=True).start()
    initialize_recommenders()
@app.post("/api/recommend/hospital")
async def recommend_hospital(request: Request):
    global hospital_recommender
    if not hospital_recommender:
        initialize_recommenders()

    total_start_time = time.time()

    data = await request.json()
    print(f"[hospital] Request data: {data}", flush=True)

    basic_info = data.get("basic_info")
    health_info = data.get("health_info")
    department = data.get("department", "내과")
    suspected_disease = data.get("suspected_disease")
    secondary_hospital = data.get("secondary_hospital", False)
    tertiary_hospital = data.get("tertiary_hospital", False)
    member_id = data.get("member_id")
    user_lat = data.get("lat")
    user_lon = data.get("lon")

    try:
        coords = address_to_coords(basic_info['address'])
        if "error" in coords:
            return JSONResponse(content={"error": coords["error"]}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    if not user_lat or not user_lon:
        user_lat = coords['lat']
        user_lon = coords['lon']

    es_results = query_elasticsearch_hosp(user_lat, user_lon, department, secondary_hospital, tertiary_hospital)
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

    recommended_hospitals = recommended_hospitals.sort_values(by=["similarity"], ascending=[False])
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

#증상, 언어 -> 병명, 질문&체크리스트
@app.post("/api/pre_question/pre_question")
async def process_symptoms(request: Request):
    try:
        data = await request.json()
        print(f"Request data: {data}", flush=True)

        symptoms = data.get('symptoms', [])
        language = data.get('language', '').upper()

        if not symptoms or not language:
            return JSONResponse(content={"error": "Both 'symptoms' and 'language' are required"}, status_code=400)

        #GPT 분석 처리
        analysis = analyze_symptoms(symptoms, language)

        final_response = {
            "department": get_department_translation(analysis["department_ko"], language),
            "possible_conditions": analysis["possible_conditions"],
            "questions_to_doctor": analysis["questions_to_doctor"],
            "symptom_checklist": analysis["symptom_checklist"]
        }

        print("final_response:", final_response, flush=True)
        return JSONResponse(content=final_response, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)