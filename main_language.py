import pymysql
import configparser
config = configparser.ConfigParser()
config.read('keys.config')

#데이터베이스 설정
DB_CONFIG = {
    "host": config['DB_INFO']['host'],
    "user": config['DB_INFO']['id'],
    "password": config['DB_INFO']['password'],
    "database": config['DB_INFO']['db'],
    "cursorclass": pymysql.cursors.DictCursor  #결과를 딕셔너리 형태로 반환
}

#언어 매핑
LANGUAGE_MAPPING = {
    1: "Korean",
    2: "English",
    3: "Vietnamese",
    4: "Chinese_Simplified",
    5: "Chinese_Traditional",
    6: "Nepali",           
    7: "Indonesian",     
    8: "Thai"              
}

#main language 조회 함수
def get_main_language(member_id):
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            #member 테이블에서 직접 language 조회
            sql = """
            SELECT language FROM member
            WHERE id = %s
            """
            cursor.execute(sql, (member_id,))
            
            result = cursor.fetchone()
            print(result)
            if result and "language" in result:
                return LANGUAGE_MAPPING.get(result["language"], "Unknown")#매핑되지 않으면 Unknown 반환
    except Exception as e:
        print(f"데이터베이스 조회 오류: {e}")
        return "Unknown"
    finally:
        connection.close()