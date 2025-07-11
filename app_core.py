from hosp_utils.recommendation import HospitalRecommender
from pharm_utils.recommendation import PharmacyRecommender
from medicine_rag_utils.drug_rag_manager import DrugRAGManager
import configparser
import mysql.connector
import pandas as pd
import schedule
import threading
import time
import os
from datetime import datetime, timedelta

def get_db_connection():
    config = configparser.ConfigParser()
    # keys.config 파일 경로 설정 (프로젝트 루트 기준)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keys.config')
    config.read(config_path)
    return mysql.connector.connect(
        host=config['DB_INFO']['host'],
        user=config['DB_INFO']['id'],
        password=config['DB_INFO']['password'],
        database=config['DB_INFO']['db']
    )

# 추천 모델 인스턴스 (전역 변수로 관리)
hospital_recommender = None
pharmacy_recommender = None
drug_rag_manager = None

def initialize_recommenders():
    global hospital_recommender, pharmacy_recommender, drug_rag_manager
    db_connection = get_db_connection()
    hospital_recommender = HospitalRecommender(db_connection=db_connection)
    pharmacy_recommender = PharmacyRecommender(db_connection=db_connection)
    drug_rag_manager = DrugRAGManager()
    drug_rag_manager.initialize_rag_system()

def retrain_models():
    global hospital_recommender, pharmacy_recommender
    query = """
    SELECT DISTINCT member_id 
    FROM (
        SELECT member_id, created_at FROM selected_hp
        UNION
        SELECT member_id, created_at FROM selected_ph
    ) combined
    WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
    """
    try:
        db_connection = get_db_connection()
        member_ids = pd.read_sql(query, db_connection)['member_id'].tolist()
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
    schedule.every().day.at("03:00").do(retrain_models)
    schedule.every().sunday.at("02:00").do(update_drug_data)
    while True:
        schedule.run_pending()
        time.sleep(60)

def update_drug_data():
    """약품 데이터 자동 업데이트 (매주 일요일 새벽 2시)"""
    try:
        global drug_rag_manager
        if drug_rag_manager:
            print("Starting automatic drug data update...")
            success = drug_rag_manager.initialize_rag_system(force_rebuild=True)
            if success:
                print("Automatic drug data update completed successfully")
            else:
                print("Automatic drug data update failed")
    except Exception as e:
        print(f"Error in automatic drug data update: {e}") 