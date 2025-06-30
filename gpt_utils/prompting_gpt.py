from .department_mapping import get_department_translation
import openai
import json
import configparser
from rag_utils.rag_search import search_similar_diseases

config=configparser.ConfigParser()
config.read('keys.config')
openai.api_key=config['API_KEYS']['chatgpt_api_key']
def analyze_symptoms(symptoms, exclude_possible_conditions=False):
    import openai, json, random
    
    # 입력 검증 (딕셔너리 구조)
    if not symptoms or not isinstance(symptoms, dict):
        raise ValueError("증상 정보는 비어있지 않은 딕셔너리여야 합니다.")

    # RAG 기반 유사 질병 검색 (Top-3)
    index_path = "rag_utils/trainings.index"
    meta_path = "rag_utils/trainings_meta.json"
    similar_conditions = search_similar_diseases(list(symptoms.values()), index_path, meta_path, top_k=3)
    similar_conditions_for_prompt = [
        {
            "disease": item["disease"],
            "symptoms": item["symptoms"]
        } for item in similar_conditions
    ]

    # 증상 설명 텍스트 생성 (영어)
    bodypart = symptoms.get("bodypart", "")
    selectedSign = symptoms.get("selectedSign", "")
    intensity = symptoms.get("intensity", "")
    startDate = symptoms.get("startDate", "")
    duration = symptoms.get("duration", "")
    state = symptoms.get("state", "")
    additional = symptoms.get("additional", "")
    symptom_description = f"Body part: {bodypart}\nSymptom: {selectedSign}\nIntensity: {intensity}\nStart date: {startDate}\nDuration: {duration}\nState: {state}"
    if additional:
        symptom_description += f"\nAdditional: {additional}"

    if not (bodypart or selectedSign or intensity or startDate or duration or state or additional):
        raise ValueError("유효한 증상 정보가 없습니다.")

    base_questions={
        "KO": [
            "통증이 생기면 어떻게 해야 하나요?",
            "치료나 진료에 대해 궁금한 점이 있으면 누구에게 연락해야 하나요?",
            "제 증상의 원인이 무엇이라고 생각하시나요?",
            "이 문제를 진단하기 위해 어떤 검사를 하실 건가요?",
            "그 검사들은 얼마나 안전한가요?",
            "치료를 할 경우와 하지 않을 경우 장기적인 예후는 어떤가요?",
            "증상이 더 심해지면 제가 스스로 해야 할 일은 무엇이고, 언제 병원에 연락해야 하나요?",
            "피해야 할 음식이나 활동이 있을까요?",
            "입원이 필요한가요?"
        ]
    }
    #질명별 질문 템플릿
    condition_question_templates={
        "KO": [
            "{}(이)라면 치료는 일반적으로 얼마나 오래 걸리나요?",
            "{}(이)라면 합병증이 생길 수도 있나요?",
            "{}(이)라면 병원에 다시 와야 하는 시점은 언제인가요?",
            "{}(이)라면 보통 사람들이 잘못 이해하는 점이 있다면 알려주세요.",
            "{}일 경우 어떤 증상이 더 나타날 수 있나요?"
        ]
    }
    
    prompt=(
        "You are a medical assistant. ALL your answers MUST be in Korean. (모든 답변은 반드시 한국어로 작성하세요.)\n\n"
        "Your task is to return the following fields in JSON format:\n\n"
        "1) 'department_ko': Based on the provided symptoms, return the most relevant Korean department (진료과) name.Choose only from the following Korean departments:\n"
        "   - 가정의학과\n"
        "   - 내과\n"
        "   - 마취통증의학과\n"
        "   - 비뇨의학과\n"
        "   - 산부인과\n"
        "   - 성형외과\n"
        "   - 소아청소년과\n"
        "   - 신경과\n"
        "   - 신경외과\n"
        "   - 심장혈관흉부외과\n"
        "   - 안과\n"
        "   - 영상의학과\n"
        "   - 예방의학과\n"
        "   - 외과\n"
        "   - 이비인후과\n"
        "   - 재활의학과\n"
        "   - 정신건강의학과\n"
        "   - 정형외과\n"
        "   - 치의과\n"
        "   - 피부과\n"
        "   - 한방과\n\n"\n"
        "2) 'possible_conditions': A list of objects, each with a 'condition' field containing the disease name in Korean. Use the following similar conditions as reference:\n"
        f"{json.dumps(similar_conditions_for_prompt, ensure_ascii=False, indent=2)}\n\n"
        "3) 'questions_to_doctor': Leave this field as an empty list []. The questions will be filled in by the application logic later.\n"
        "4) 'symptom_checklist': A list of objects. Each object must contain:\n"
        "   - 'condition_ko': the condition name in Korean\n"
        "   - 'symptoms': a list of symptom names in Korean\n\n"
        "Use only medically relevant conditions based on the symptoms. Use formal medical language.\n\n"
        f"Respond ONLY with valid JSON in the following structure:\n"
        "{{\n"
        '  "department_ko": "정형외과", \n'
        '  "possible_conditions": [ {{"condition": "..."}} ],\n'
        '  "questions_to_doctor": [ {{"KO": "..."}} ],\n'
        '  "symptom_checklist": {{\n'
        '    "무릎 관절염": {{\n'
        '      "symptoms": [ "..." ]\n'
        "    }}\n"
        "  }}\n"
        "}}\n\n"
        "Respond ONLY with valid JSON. Do NOT include any explanation or formatting. No markdown.\n\n"
    )

    response=openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Symptoms: {symptom_description}"
            }
        ],
        temperature=0.3
    )

    result=json.loads(response.choices[0].message.content.strip())
    questions_to_doctor=[]

    base_count=min(3, len(base_questions["KO"]))
    condition_count=min(2, len(result["possible_conditions"]))

    #base 질문 처리
    for q in random.sample(range(len(base_questions["KO"])), base_count):
        questions_to_doctor.append({"KO": base_questions["KO"][q]})

    #condition 질문 처리
    chosen_idxs=random.sample(range(len(result["possible_conditions"])), condition_count)
    template_idxs=random.sample(range(len(condition_question_templates["KO"])), 2)

    for i, cond_idx in enumerate(chosen_idxs):
        cond=result["possible_conditions"][cond_idx]["condition"]
        t_idx=template_idxs[i]
        questions_to_doctor.append({"KO": condition_question_templates["KO"][t_idx].format(cond["KO"])})

    result["questions_to_doctor"]=questions_to_doctor

    if exclude_possible_conditions and "possible_conditions" in result:
        del result["possible_conditions"]

    return result

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
        language (str): 사용자 언어 ('KO', 'EN', 'VI', 'ZH_CN', 'ZH_TW')
    
    Returns:
        list[str]: 3개의 형용사 리스트
    """
    
    # 언어별 프롬프트 템플릿
    language_prompts = {
        "KO": f"'{body_part}'에 대해 일반적으로 사용되는 증상 형용사 3개를 한국어로만 반환해주세요. 예시: '아파요', '부어요', '따가워요'와 같은 형식으로 답변해주세요. JSON 배열 형태로만 응답하세요.",
        "EN": f"Please return 3 common symptom adjectives for '{body_part}' in English only. Examples: 'hurts', 'swollen', 'tingling'. Respond only in JSON array format.",
        "VI": f"Vui lòng trả về 3 tính từ triệu chứng phổ biến cho '{body_part}' bằng tiếng Việt. Ví dụ: 'đau', 'sưng', 'ngứa'. Chỉ trả lời dưới dạng mảng JSON.",
        "ZH_CN": f"请为'{body_part}'返回3个常见症状形容词，仅用中文。例如：'疼'、'肿'、'痒'。仅以JSON数组格式回复。",
        "ZH_TW": f"請為'{body_part}'返回3個常見症狀形容詞，僅用繁體中文。例如：'疼'、'腫'、'癢'。僅以JSON陣列格式回覆。"
    }
    
    prompt = language_prompts.get(language.upper(), language_prompts["EN"])
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a medical assistant that provides symptom adjectives for body parts. Respond only with valid JSON array containing exactly 3 adjectives."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        
        # JSON 파싱
        import json
        adjectives = json.loads(result)
        
        # 3개 형용사가 아닌 경우 처리
        if not isinstance(adjectives, list) or len(adjectives) != 3:
            # 기본 형용사 반환
            default_adjectives = {
                "KO": ["아파요", "부어요", "따가워요"],
                "EN": ["hurts", "swollen", "tingling"],
                "VI": ["đau", "sưng", "ngứa"],
                "ZH_CN": ["疼", "肿", "痒"],
                "ZH_TW": ["疼", "腫", "癢"]
            }
            return default_adjectives.get(language.upper(), default_adjectives["EN"])
        
        return adjectives
        
    except Exception as e:
        print(f"Error in get_body_part_adjectives: {e}")
        # 에러 시 기본 형용사 반환
        default_adjectives = {
            "KO": ["아파요", "부어요", "따가워요"],
            "EN": ["hurts", "swollen", "tingling"],
            "VI": ["đau", "sưng", "ngứa"],
            "ZH_CN": ["疼", "肿", "痒"],
            "ZH_TW": ["疼", "腫", "癢"]
        }
        return default_adjectives.get(language.upper(), default_adjectives["EN"])

def select_department_for_symptoms(symptoms: dict) -> str:
    """
    Given symptoms, use GPT to select the most relevant department (진료과) in Korean.
    Returns the department name in Korean.
    """
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
    duration = symptoms.get("duration", "")
    state = symptoms.get("state", "")
    additional = symptoms.get("additional", "")
    symptom_description = f"Body part: {bodypart}\nSymptom: {selectedSign}\nIntensity: {intensity}\nStart date: {startDate}\nDuration: {duration}\nState: {state}"
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
    # Only keep the department if it's in the list
    if department_ko not in department_list:
        # fallback: just return 내과
        return "내과"
    return department_ko


def recommend_questions_to_doctor(symptoms: dict) -> list:
    """
    Recommend 4 questions to ask the doctor, in Korean, based on the symptoms.
    Returns a list of 4 strings (Korean).
    """
    import random
    # Use the same base_questions and condition_question_templates as in analyze_symptoms
    base_questions = [
        "통증이 생기면 어떻게 해야 하나요?",
        "치료나 진료에 대해 궁금한 점이 있으면 누구에게 연락해야 하나요?",
        "제 증상의 원인이 무엇이라고 생각하시나요?",
        "이 문제를 진단하기 위해 어떤 검사를 하실 건가요?",
        "그 검사들은 얼마나 안전한가요?",
        "치료를 할 경우와 하지 않을 경우 장기적인 예후는 어떤가요?",
        "증상이 더 심해지면 제가 스스로 해야 할 일은 무엇이고, 언제 병원에 연락해야 하나요?",
        "피해야 할 음식이나 활동이 있을까요?",
        "입원이 필요한가요?"
    ]
    condition_question_templates = [
        "{}(이)라면 치료는 일반적으로 얼마나 오래 걸리나요?",
        "{}(이)라면 합병증이 생길 수도 있나요?",
        "{}(이)라면 병원에 다시 와야 하는 시점은 언제인가요?",
        "{}(이)라면 보통 사람들이 잘못 이해하는 점이 있다면 알려주세요.",
        "{}일 경우 어떤 증상이 더 나타날 수 있나요?"
    ]
    # Pick 2 base questions randomly
    base_selected = random.sample(base_questions, 2)
    # For condition-based questions, use selectedSign as the condition
    condition = symptoms.get("selectedSign", "이 증상")
    template_selected = random.sample(condition_question_templates, 2)
    condition_questions = [tpl.format(condition) for tpl in template_selected]
    return base_selected + condition_questions

def translate_text(text, target_language="en"):
    #입력 텍스트를 target_language로 번역
    #욕설, 공격적인 표현을 정제

    language_map={
        "ko": "Korean",
        "en": "English",
        "vi": "Vietnamese",
        "zh_cn": "Chinese(Simplified)",
        "zh_tw": "Chinese(Traditional)"
    }

    target_lang_full=language_map.get(target_language.lower(), "English")

    prompt=(
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