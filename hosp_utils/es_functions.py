def query_elasticsearch_hosp(user_lat, user_lon, department=None, primary_hospital=False, secondary_hospital=False, tertiary_hospital=False):
    from elasticsearch import Elasticsearch
    import configparser
    config=configparser.ConfigParser()
    config.read('keys.config')
    
    # department가 리스트인 경우 문자열로 변환
    if isinstance(department, list):
        department = ", ".join(department)
    
    #Elasticsearch 클라이언트 설정
    es=Elasticsearch(
        hosts=[config['ES_INFO']['host']],
        basic_auth=(config['ES_INFO']['username'], config['ES_INFO']['password']),
        verify_certs=False
    )

    #병원 유형 필터 구성
    must_clcdnm=[]
    
    # 모든 조건이 true이거나 false이면 필터링 없이 전체 조회
    if (primary_hospital and secondary_hospital and tertiary_hospital) or (not primary_hospital and not secondary_hospital and not tertiary_hospital):
        must_clcdnm = []
    else:
        # 선택된 병원 유형들의 clcdnm을 하나의 리스트로 모음
        selected_types = []
        
        if primary_hospital:
            # 1차 의료기관: 의원, 의료원, 보건소 등
            primary_types = ["의원", "치과의원", "요양병원", "정신병원", "한의원", "치과병원", "한방병원", "의료원", "보건소", "보건지소", "진료소", "보건의료원", "보건진료소"]
            selected_types.extend(primary_types)
        if secondary_hospital:
            # 2차 의료기관: 병원, 종합병원
            secondary_types = ["병원", "종합병원"]
            selected_types.extend(secondary_types)
        if tertiary_hospital:
            # 3차 의료기관: 상급종합병원
            tertiary_types = ["상급종합"]
            selected_types.extend(tertiary_types)
        
        # 선택된 유형들을 하나의 terms 쿼리로 처리
        if selected_types:
            must_clcdnm = [{"terms": {"clcdnm": selected_types}}]

    #Elasticsearch 쿼리 구성
    must_queries=[]
    
    # 여러 진료과 지원 (쉼표로 구분된 진료과 처리)
    if department:
        departments = [dept.strip() for dept in department.split(',')]
        department_queries = []
        
        for dept in departments:
            #특수한 department 케이스 처리
            if dept == "치의과":
                dental_departments=[
                    "치과", "구강악안면외과", "치과보철과", "치과교정과", "소아치과",
                    "치주과", "치과보존과", "구강내과", "영상치의학과", "구강병리과",
                    "예방치과", "통합치의학과"
                ]
                department_queries.append({"terms": {"dgsbjt": dental_departments}})
            elif dept == "한방과":
                oriental_departments=[
                    "한방내과", "한방부인과", "한방소아과", "한방안·이비인후·피부과",
                    "한방신경정신과", "침구과", "한방재활의학과", "사상체질과", "한방응급"
                ]
                department_queries.append({"terms": {"dgsbjt": oriental_departments}})
            else:
                #기존 department 처리
                department_queries.append({"match_phrase": {"dgsbjt": dept}})
        
        # 여러 진료과를 OR 조건으로 결합
        if len(department_queries) > 1:
            must_queries.append({"bool": {"should": department_queries}})
        else:
            must_queries.extend(department_queries)

    #메인 쿼리
    query={
        "_source": True,  #_source 필드 포함
        "query": {
            "bool": {
                "must": must_queries,
                "filter": [
                    {"geo_distance": {"distance": "100km", "location": {"lat": user_lat, "lon": user_lon}}}
                ]
            }
        },
        "sort": [
            {"_geo_distance": {"location": {"lat": user_lat, "lon": user_lon}, "order": "asc", "unit": "km"}}
        ],
        "script_fields": {  #거리 값을 반환하도록 스크립트 필드 추가
            "es_distance_in_km": {
                "script": {
                    "source": "doc['location'].arcDistance(params.lat, params.lon) / 1000",  #km 거리 반환
                    "params": {"lat": user_lat, "lon": user_lon}
                }
            }
        },
        "size": 30  #최대 30개 결과 제한
    }

    #병원 유형 필터 추가
    if must_clcdnm:
        query["query"]["bool"]["must"].extend(must_clcdnm)
    
    #Elasticsearch 검색 실행
    response=es.search(
        index="hospital_records_v3",
        body=query
    )
    
    return response

def filtering_hosp(results):
    """
    Elasticsearch 결과 필터링
    """
    filtered_results=[]
    for hit in results['hits']['hits']:
        source=hit['_source']

        es_distance_in_km=hit.get("fields", {}).get("es_distance_in_km", [None])[0] #script_fields 값 읽기
        filtered_results.append({
            "id": source.get("id"),
            "name": source.get("yadmnm"),
            "address": source.get("addr"),
            "telephone": source.get("telno"),
            "department": source.get("dgsbjt"),
            "latitude": source.get("ypos"),
            "longitude": source.get("xpos"),
            "es_distance_in_km": es_distance_in_km, 
            "sidocdnm": source.get("sidocdnm"),
            "sggucdnm": source.get("sggucdnm"),
            "emdongnm": source.get("emdongnm"),
            "clcdnm": source.get("clcdnm"),
            "location": source.get("location"),
            "url": source.get("hospurl"),
            "sort_score": hit.get("sort", [None])[0]  #정렬 기준 추가
        })
    return filtered_results