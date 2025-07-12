import openai
import json
import configparser
import os
from rag_utils.rag_search import search_similar_diseases
import time

#keys.config 파일 경로 설정 (프로젝트 루트 기준)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
config_path = os.path.join(project_root, 'keys.config')

config=configparser.ConfigParser()
config.read(config_path)
openai.api_key=config['API_KEYS']['chatgpt_api_key']

# 언어 정보 통합 딕셔너리
LANGUAGES = {
    "KO": {
        "name": "한국어",
        "adjectives": ["아파요", "부었어요", "따가워요"],
        "prompt": "'{body_part}'에 대해 일반적으로 사용되는 증상 형용사 3개를 한국어로만 반환해주세요. 반드시 '해요체' 형태로 답변해주세요. 예시: '아파요', '부었어요', '뻣뻣해요', '저려요', '따가워요', '시큰거려요'와 같은 형식으로 답변해주세요. JSON 배열 형태로만 응답하세요.",
    },
    "EN": {
        "name": "English",
        "adjectives": ["hurts", "swollen", "tingling"],
        "prompt": "Please return 3 common symptom adjectives for '{body_part}' in English only. Examples: 'hurts', 'swollen', 'tingling'. Respond only in JSON array format.",
    },
    "VI": {
        "name": "Vietnamese",
        "adjectives": ["đau", "sưng", "ngứa"],
        "prompt": "Vui lòng trả về 3 tính từ triệu chứng phổ biến cho '{body_part}' bằng tiếng Việt. Ví dụ: 'đau', 'sưng', 'ngứa'. Chỉ trả lời dưới dạng mảng JSON.",
    },
    "ZH_CN": {
        "name": "Chinese (Simplified)",
        "adjectives": ["疼", "肿", "痒"],
        "prompt": "请为'{body_part}'返回3个常见症状形容词，仅用中文。例如：'疼'、'肿'、'痒'。仅以JSON数组格式回复。",
    },
    "ZH_TW": {
        "name": "Chinese (Traditional)",
        "adjectives": ["疼", "腫", "癢"],
        "prompt": "請為'{body_part}'返回3個常見症狀形容詞，僅用繁體中文。例如：'疼'、'腫'、'癢'。僅以JSON陣列格式回覆。",
    },
    "NE": {
        "name": "Nepali",
        "adjectives": ["दुख्छ", "सुन्निन्छ", "खुजली हुन्छ"],
        "prompt": "कृपया '{body_part}' का लागि ३ सामान्य लक्षण विशेषणहरू नेपालीमा मात्र फर्काउनुहोस्। उदाहरण: 'दुख्छ', 'सुन्निन्छ', 'खुजली हुन्छ'। केवल JSON array मा जवाफ दिनुहोस्।",
    },
    "ID": {
        "name": "Indonesian",
        "adjectives": ["sakit", "bengkak", "gatal"],
        "prompt": "Silakan kembalikan 3 kata sifat gejala umum untuk '{body_part}' hanya dalam bahasa Indonesia. Contoh: 'sakit', 'bengkak', 'gatal'. Jawab hanya dalam format array JSON.",
    },
    "TH": {
        "name": "Thai",
        "adjectives": ["ปวด", "บวม", "คัน"],
        "prompt": "โปรดส่งคืนคำคุณศัพท์อาการทั่วไป 3 คำสำหรับ '{body_part}' เป็นภาษาไทยเท่านั้น ตัวอย่าง: 'ปวด', 'บวม', 'คัน' ตอบกลับเป็น JSON array เท่านั้น",
    },
}

def analyze_symptoms_with_summary(symptoms, patient_info=None, language="KO"):
    """
    증상 분석과 요약을 한 번에 처리합니다.
    기존 analyze_symptoms, summarize_symptom_input, summarize_symptom_keywords를 통합합니다.
    """
    import openai, json
    t_total = time.time()
    
    #입력 검증 (딕셔너리 구조)
    if not symptoms or not isinstance(symptoms, dict):
        raise ValueError("증상 정보는 비어있지 않은 딕셔너리여야 합니다.")

    t0 = time.time()
    #RAG 기반 유사 질병 검색 (Top-3)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    rag_utils_dir = os.path.join(project_root, 'rag_utils')
    
    # IVF 인덱스 파일 우선 확인
    ivf_index_path = os.path.join(rag_utils_dir, "combined_ivf.index")
    meta_path = os.path.join(rag_utils_dir, "combined_meta.json")
    
    # IVF 인덱스가 있으면 사용, 없으면 기존 인덱스 사용
    if os.path.exists(ivf_index_path):
        index_path = ivf_index_path
        print("[IVF STATUS] Using IVF index for disease search")
    else:
        index_path = os.path.join(rag_utils_dir, "combined.index")
        print("[IVF STATUS] Using standard index for disease search (no IVF index available)")
    
    similar_conditions = search_similar_diseases(list(symptoms.values()), index_path, meta_path, top_k=3)
    print(f"[latency][RAG] 유사 질병 검색: {time.time() - t0:.3f}초", flush=True)
    t0 = time.time()
    similar_conditions_for_prompt = [
        {
            "disease": item["disease"],
            "symptoms": item["symptoms"]
        } for item in similar_conditions
    ]

    #증상 설명 텍스트 생성 (영어)
    bodypart = symptoms.get("bodypart", "")
    selectedSign = symptoms.get("selectedSign", "")
    intensity = symptoms.get("intensity", "")
    startDate = symptoms.get("startDate", "")
    duration = symptoms.get("duration", "")
    state = symptoms.get("state", "")
    additional = symptoms.get("additional", "")
    symptom_description = f"Body part: {bodypart}\nSymptom: {selectedSign}\nIntensity: {intensity}\nStart date: {startDate}"
    if duration:
        symptom_description += f"\nDuration: {duration}"
    if state:
        symptom_description += f"\nState: {state}"
    if additional:
        symptom_description += f"\nAdditional: {additional}"

    #환자 정보 텍스트 생성
    patient_info_text = ""
    if patient_info:
        gender = patient_info.get("gender", "")
        age = patient_info.get("age", "")
        allergy = patient_info.get("allergy", "")
        family_history = patient_info.get("familyHistory", "")
        now_medicine = patient_info.get("nowMedicine", "")
        past_history = patient_info.get("pastHistory", "")
        
        patient_info_text = f"\nPatient Information:\nGender: {gender}\nAge: {age}\nAllergy: {allergy}\nFamily History: {family_history}\nCurrent Medicine: {now_medicine}\nPast Medical History: {past_history}"

    if not (bodypart or selectedSign or intensity or startDate or duration or state or additional):
        raise ValueError("유효한 증상 정보가 없습니다.")
    
    # 증상 요약 텍스트 생성
    symptom_summary_text = f"환자 정보: {patient_info.get('gender', '')}, {patient_info.get('age', '')}세\n신체 부위: {bodypart}\n증상 설명: {selectedSign}\n강도: {intensity}\n시작일: {startDate}\n지속 기간: {duration}\n상태: {state}"
    if patient_info.get('allergy'):
        symptom_summary_text += f"\n알레르기: {patient_info['allergy']}"
    if patient_info.get('nowMedicine'):
        symptom_summary_text += f"\n복용약: {patient_info['nowMedicine']}"
    if patient_info.get('pastHistory'):
        symptom_summary_text += f"\n과거력: {patient_info['pastHistory']}"
    if patient_info.get('familyHistory'):
        symptom_summary_text += f"\n가족력: {patient_info['familyHistory']}"
    if additional:
        symptom_summary_text += f"\n추가 설명: {additional}"
    
    # 증상 키워드 생성
    summary_keywords = []
    if patient_info.get('gender'):
        summary_keywords.append(str(patient_info['gender']))
    if patient_info.get('age'):
        summary_keywords.append(f"{patient_info['age']}세")
    if patient_info.get('allergy'):
        summary_keywords.append(f"알레르기:{patient_info['allergy']}")
    if patient_info.get('nowMedicine'):
        summary_keywords.append(f"복용약:{patient_info['nowMedicine']}")
    if patient_info.get('pastHistory'):
        summary_keywords.append(f"과거력:{patient_info['pastHistory']}")
    if patient_info.get('familyHistory'):
        summary_keywords.append(f"가족력:{patient_info['familyHistory']}")
    if bodypart:
        summary_keywords.append(str(bodypart))
    if selectedSign:
        summary_keywords.append(str(selectedSign))
    if duration:
        summary_keywords.append(str(duration))
    if state:
        summary_keywords.append(str(state))
    if additional:
        summary_keywords.append(str(additional))
    
    t1 = time.time()
    # 통합 프롬프트 생성
    prompt = f"""당신은 의학 전문가입니다. 다음 증상 정보를 분석하여 진료과, 증상 요약, 키워드 요약을 한 번에 제공해주세요.

환자 정보:
{patient_info_text}

증상 정보:
{symptom_description}

유사 질병 참고:
{json.dumps(similar_conditions_for_prompt, ensure_ascii=False, indent=2)}

다음 JSON 형식으로 응답해주세요:
{{
    "department_ko": "진료과명",
    "symptom_summary_ko": "증상 요약 (한국어)",
    "symptom_summary_trans": "증상 요약 (번역)",
    "summary_keywords_ko": "키워드 요약 (한국어)",
    "summary_keywords_trans": "키워드 요약 (번역)"
}}

진료과는 다음 중에서 선택하세요:
가정의학과, 내과, 마취통증의학과, 비뇨의학과, 산부인과, 성형외과, 소아청소년과, 신경과, 신경외과, 심장혈관흉부외과, 안과, 영상의학과, 예방의학과, 외과, 이비인후과, 재활의학과, 정신건강의학과, 정형외과, 치의과, 피부과, 한방과

모든 답변은 반드시 한국어로 작성하세요."""

    print(f"[latency][RAG] 프롬프트 생성: {time.time() - t1:.3f}초", flush=True)
    user_content = f"Symptoms: {symptom_description}{patient_info_text}"
    t2 = time.time()
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": user_content
            }
        ],
        temperature=0.3
    )
    print(f"[latency][RAG] LLM 호출: {time.time() - t2:.3f}초", flush=True)
    result = json.loads(response.choices[0].message.content.strip())
    print(f"[latency][RAG] 전체 소요 시간: {time.time() - t_total:.3f}초", flush=True)
    
    # 결과를 분리하여 반환
    department_ko = result.get("department_ko", "")
    symptom_summary_ko = result.get("symptom_summary_ko", "")
    symptom_summary_trans = result.get("symptom_summary_trans", "")
    summary_keywords_ko = result.get("summary_keywords_ko", "")
    summary_keywords_trans = result.get("summary_keywords_trans", "")
    
    # 언어에 따른 요약 텍스트 선택
    if language.upper() != "KO":
        symptom_summary = f"{symptom_summary_ko} ({symptom_summary_trans})"
        summary_text = f"{summary_keywords_ko} ({summary_keywords_trans})"
    else:
        symptom_summary = symptom_summary_ko
        summary_text = summary_keywords_ko
    
    return {
        "department_ko": department_ko,
        "symptom_summary": symptom_summary,
        "summary_text": summary_text
    }

def romanize_korean_names(names: list[str]) -> dict:
    #병원명이나 약국명을 GPT를 사용해 음독(로마자 표기)
    import openai
    import json

    try:
        system_prompt=(
            "You are a Korean language expert. Convert the following Korean medical facility names into Romanized Korean "
            "using proper spacing. Always separate the medical suffix at the end like '병원', '의원', '약국', '한의원'.\n"
            "Return the result as a JSON dictionary where each key is the original name and the value is the Romanized name. "
            "No explanation, only valid JSON.\n"
            "Example:\n"
            "{\n"
            "  \"강현우비뇨기과의원\": \"Kanghyunu Binyogigwa Uiwon\",\n"
            "  \"해림온누리약국\": \"Haerim Onnuri Yakguk\"\n"
            "}"
        )

        name_list_text="\n".join(f"- {name}" for name in names)

        user_prompt=f"Romanize the following names:\n{name_list_text}"

        response=openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        result_text=response.choices[0].message.content.strip()
        result=json.loads(result_text)
        return result

    except Exception as e:
        print(f"Romanization Error: {e}")
        return {}

def get_body_part_adjectives(body_part: str, language: str) -> list[str]:
    """
    body part를 받아서 해당 부위에 대한 3개의 형용사를 반환합니다.
    
    Args:
        body_part (str): 신체 부위 (예: '무릎', '머리', '배')
        language (str): 사용자 언어 ('KO', 'EN', 'VI', 'ZH_CN', 'ZH_TW', 'NE', 'ID', 'TH')
    
    Returns:
        list[str]: 3개의 형용사 리스트
    """
    
    lang = LANGUAGES.get(language.upper(), LANGUAGES["EN"])
    prompt = lang["prompt"].format(body_part=body_part)
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a medical assistant that provides symptom adjectives for body parts. For Korean language, always use 해요체 (polite informal form ending in -요). Respond only with valid JSON array containing exactly 3 adjectives."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        
        #JSON 파싱
        import json
        adjectives = json.loads(result)
        
        #3개 형용사가 아닌 경우 처리
        if not isinstance(adjectives, list) or len(adjectives) != 3:
            return lang["adjectives"]
        
        return adjectives
        
    except Exception as e:
        print(f"Error in get_body_part_adjectives: {e}")
        return lang["adjectives"]

def select_department_for_symptoms(symptoms: dict) -> str:

    import openai
    import json
    
    department_list = [
        "가정의학과", "내과", "마취통증의학과", "비뇨의학과", "산부인과", "성형외과", "소아청소년과",
        "신경과", "신경외과", "심장혈관흉부외과", "안과", "영상의학과", "예방의학과", "외과",
        "이비인후과", "재활의학과", "정신건강의학과", "정형외과", "치의과", "피부과", "한방과"
    ]
    
    bodypart = symptoms.get("bodypart", "")
    selectedSign = symptoms.get("selectedSign", "")
    intensity = symptoms.get("intensity", "")
    startDate = symptoms.get("startDate", "")
    additional = symptoms.get("additional", "")
    symptom_description = f"Body part: {bodypart}\nSymptom: {selectedSign}\nIntensity: {intensity}\nStart date: {startDate}"
    if additional:
        symptom_description += f"\nAdditional: {additional}"
    
    prompt = (
        "You are a medical assistant. ALL your answers MUST be in Korean. (모든 답변은 반드시 한국어로 작성하세요.)\n"
        "Given the following symptom information, select the most relevant department (진료과) from the list below.\n"
        "Return ONLY the department name in Korean, no explanation, no JSON, no markdown.\n"
        "Department list: " + ", ".join(department_list) + "\n"
        f"Symptoms: {symptom_description}"
    )
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0.2
    )
    department_ko = response.choices[0].message.content.strip()
    #Only keep the department if it's in the list
    if department_ko not in department_list:
        #fallback: just return 내과
        return "내과"
    return department_ko


def recommend_questions_to_doctor(symptoms: dict) -> list:
    """
    Recommend 4 questions to ask the doctor, in Korean, based on the symptoms.
    Returns a list of 4 strings (Korean).
    """
    import openai
    import json
    
    #증상 정보 구성
    bodypart = symptoms.get("bodypart", "")
    selectedSign = symptoms.get("selectedSign", "")
    intensity = symptoms.get("intensity", "")
    startDate = symptoms.get("startDate", "")
    duration = symptoms.get("duration", "")
    state = symptoms.get("state", "")
    additional = symptoms.get("additional", "")
    fallback_questions = [
        "제 증상의 원인이 무엇이라고 생각하시나요?",
        "이 문제를 진단하기 위해 어떤 검사를 하실 건가요?",
        "치료는 일반적으로 얼마나 오래 걸리나요?",
        "증상이 더 심해지면 어떻게 해야 하나요?"
    ]
    
    symptom_description = f"Body part: {bodypart}\nSymptom: {selectedSign}\nIntensity: {intensity}\nStart date: {startDate}"
    if duration:
        symptom_description += f"\nDuration: {duration}"
    if state:
        symptom_description += f"\nState: {state}"
    if additional:
        symptom_description += f"\nAdditional: {additional}"
    
    prompt = (
        "You are a medical assistant. ALL your answers MUST be in Korean. (모든 답변은 반드시 한국어로 작성하세요.)\n\n"
        "Based on the provided symptom information, generate exactly 4 relevant questions that a patient should ask their doctor.\n"
        "The questions should be:\n"
        "1. Specific to the symptoms described\n"
        "2. Natural and conversational in Korean\n"
        "3. Appropriate for a patient to ask their doctor\n"
        "4. Cover different aspects: diagnosis, treatment, prevention, and follow-up\n\n"
        "Consider the following when generating questions:\n"
        "- If the symptom is described as a sentence, create questions that naturally reference the symptom\n"
        "- If the symptom is a single word, you can use it directly in questions\n"
        "- Make questions sound natural and not awkward\n"
        "- Use polite Korean form (해요체)\n\n"
        f"Symptom information:\n{symptom_description}\n\n"
        "Return ONLY a JSON array with exactly 4 questions in Korean:\n"
        '["질문1", "질문2", "질문3", "질문4"]\n\n'
        "Respond ONLY with valid JSON. Do NOT include any explanation or formatting."
    )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        questions = json.loads(result)
        
        #4개 질문이 아닌 경우 처리
        if not isinstance(questions, list) or len(questions) != 4:
            #기본 질문으로 fallback
            print(f"Error in recommend_questions_to_doctor: {e}")
            return fallback_questions
        
        return questions
        
    except Exception as e:
        print(f"Error in recommend_questions_to_doctor: {e}")
        return fallback_questions

def translate_text(text, target_language="en"):
    lang = LANGUAGES.get(target_language.upper(), LANGUAGES["EN"])
    target_lang_full = lang["name"]
    prompt = (
        f"You are a professional medical translator. "
        f"Translate the given short text to {target_lang_full} if it is not already in that language. "
        f"Do NOT explain what language the input is in. "
        f"DO NOT include any introductory phrases such as 'Here is the translation' or 'The input means...'. "
        f"Only return the translated result — no commentary, no explanation. "
        f"If the text is already in {target_lang_full}, return it as is (cleaned). "
        f"If the text contains any vulgar or offensive expressions, rewrite them politely. "
    )
    try:
        response=openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GPT translation error]: {e}")
        return text  #fallback: 원본 반환


def recommend_drug_llm_response(drug_candidates: list, symptom: str, patient_info: dict, language: str = "KO") -> dict:
    """
    FAISS로 추출한 drug 후보(메타데이터)와 증상, 환자 정보를 받아,
    영어 프롬프트로 concise JSON만 반환하도록 LLM에 요청하는 함수.
    """
    import openai
    import json
    language_map = {
        "KO": "Korean",
        "EN": "English",
        "VI": "Vietnamese",
        "ZH_CN": "Chinese (Simplified)",
        "ZH_TW": "Chinese (Traditional)",
        "NE": "Nepali",
        "ID": "Indonesian",
        "TH": "Thai"
    }
    lang_str = language_map.get(language.upper(), "Korean")
    candidate_summaries = []
    for drug in drug_candidates:
        summary = f"Name: {drug.get('itemName', '')}, Purpose: {drug.get('efcyQesitm', '')}, Image: {drug.get('itemImage', '')}"
        candidate_summaries.append(summary)
    candidates_text = "\n".join(candidate_summaries)

    pi = patient_info or {}
    pi_parts = []
    
    # 환자 정보를 한국어로 번역
    if pi.get("allergy"):
        try:
            allergy_translated = translate_text(pi['allergy'], "ko") if pi['allergy'] else ""
        except:
            allergy_translated = pi['allergy']  # 번역 실패 시 원본 사용
        pi_parts.append(f"알레르기:{allergy_translated}")
    if pi.get("familyHistory"):
        try:
            family_translated = translate_text(pi['familyHistory'], "ko") if pi['familyHistory'] else ""
        except:
            family_translated = pi['familyHistory']  # 번역 실패 시 원본 사용
        pi_parts.append(f"가족력:{family_translated}")
    if pi.get("nowMedicine"):
        try:
            medicine_translated = translate_text(pi['nowMedicine'], "ko") if pi['nowMedicine'] else ""
        except:
            medicine_translated = pi['nowMedicine']  # 번역 실패 시 원본 사용
        pi_parts.append(f"현재복용약:{medicine_translated}")
    if pi.get("pastHistory"):
        try:
            history_translated = translate_text(pi['pastHistory'], "ko") if pi['pastHistory'] else ""
        except:
            history_translated = pi['pastHistory']  # 번역 실패 시 원본 사용
        pi_parts.append(f"과거병력:{history_translated}")
    
    pi_summary = ", ".join(pi_parts)

    prompt = (
        f"You are a medical AI assistant. Your answers must be in Korean.\n"
        f"First, extract a concise symptom keyword or short phrase (2-10 characters, in Korean) that best summarizes the following symptom input.\n"
        f"You MUST recommend ONLY from the following drug candidates list. Do NOT recommend anything that is not in the list.\n"
        f"Then, use that keyword/phrase to generate two natural, polite (해요체) questions for a pharmacist as follows:\n"
        f"1. '[키워드]에 좋은 약' (Use the extracted keyword/phrase in place of [키워드]. Do NOT just paste the whole symptom if it is a long sentence. Write ONLY as a noun phrase. Do NOT end with a question or use '있을까요?'.)\n"
        f"2. '[키워드] 증상이 있어요. 적합한 약이 있으신가요?' (Again, use the keyword/phrase, not the whole sentence.)\n"
        f"If the symptom is a sentence, extract the main keyword or summarize it concisely for use in the questions.\n"
        f"If patient information is present (allergy, familyHistory, nowMedicine, pastHistory), add a third question: '환자에게 처방 시 다음 내용을 참고해주세요\\n{pi_summary}'\n"
        f"Return ONLY a valid JSON object with the following fields:\n"
        f"- drug_name: string (MUST be in Korean, do NOT translate, MUST be one of the drug candidates below)\n"
        f"- drug_purpose: string (MUST be a translation of the 'efcyQesitm' field of the selected drug, in {lang_str})\n"
        f"- drug_image_url: string (MUST be the 'itemImage' field of the selected drug, or empty string)\n"
        f"- pharmacist_questions: array of up to 3 questions, as described above.\n"
        f"\nDrug candidates (choose ONLY from this list!):\n{candidates_text}\n"
        f"\nExample:\n"
        f"{{\n"
        f"  \"drug_name\": \"타이레놀\",\n"
        f"  \"drug_purpose\": \"For mild to moderate pain relief and fever reduction.\",\n"
        f"  \"drug_image_url\": \"http://...\",\n"
        f"  \"pharmacist_questions\": [\"목 통증에 좋은 약 있을까요?\", \"목 통증 증상이 있어요. 적합한 약이 있으신가요?\", \"환자에게 처방 시 다음 내용을 참고해주세요\\n알레르기:penicillin\"]\n"
        f"}}\n\n"
        f"Symptom input: {symptom}\n"
        f"Patient info: {json.dumps(patient_info, ensure_ascii=False)}\n"
        f"Answer in JSON, following the above rules. Do NOT include any explanation or extra text."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        return json.loads(result)
    except Exception as e:
        print(f"[recommend_drug_llm_response error]: {e}")
        return {}