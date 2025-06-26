import openai
import configparser
import json
import boto3
import tempfile
import openai
import requests
from langdetect import detect

config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']

    
def download_audio_from_s3_presigned_url(presigned_url):
    """
    S3 presigned URL에서 오디오 파일을 다운로드하여 임시 파일로 저장
    :param presigned_url: S3 presigned URL
    :return: 임시 파일 경로
    """
    try:
        response = requests.get(presigned_url, stream=True)
        response.raise_for_status()
        
        #임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file_path = temp_file.name
        
        #파일에 데이터 쓰기
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        
        temp_file.close()
        return temp_file_path
        
    except Exception as e:
        print(f"S3에서 오디오 파일 다운로드 실패: {e}")
        raise e
    
def transcribe_audio(file_path):
    """
    OpenAI Whisper를 사용하여 음성을 텍스트로 변환
    :param file_path: 변환할 오디오 파일 경로
    :return: 변환된 텍스트
    """
    with open(file_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            prompt="이 오디오는 환자가 얘기하거나 의사가 얘기하는 내용입니다. 이를 고려해서 transcribe 해주세요."
        )
    return transcript.text

LANGUAGE_MAP = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese(Simplified)",
    "zh-tw": "Chinese(Traditional)"
}



def translate_text_simple(text, tag, main_language="English"):
    """
    단순화된 번역 함수
    - 환자(tag=0): 한국어로 번역
    - 의료진(tag=1): main_language로 번역
    :param text: 번역할 텍스트
    :param tag: 화자 구분 (0: 환자, 1: 의료진)
    :param main_language: 의료진의 주요 언어 (기본값: English)
    :return: 번역 결과 딕셔너리
    """
    if tag == 0:
        target_language = "Korean"
    elif tag == 1:
        target_language = main_language
    else:
        #기본값
        target_language = "Korean"
    
    prompt = f"""
    Translate the following text to {target_language}.
    Text: {text}
    
    Respond with only the translated text, no additional formatting or explanations.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        translated_text = response.choices[0].message.content.strip()
        
        return {
            "translations": {
                target_language: {"text": translated_text}
            }
        }
    except Exception as e:
        print(f"번역 실패: {e}")
        return {
            "translations": {
                target_language: {"text": text}  #번역 실패 시 원문 반환
            }
        }

def detect_language_simple(text):
    """
    텍스트의 언어를 감지하는 단순화된 함수
    :param text: 감지할 텍스트
    :return: 감지된 언어명
    """
    try:
        lang_code = detect(text)
        return LANGUAGE_MAP.get(lang_code, "Unknown")
    except Exception as e:
        print(f"언어 감지 실패: {e}")
        return "Unknown"

#AWS S3 설정
S3_BUCKET_NAME = config['S3_INFO']['BUCKET_NAME']
s3_client = boto3.client("s3",
                        aws_access_key_id=config['S3_INFO']['ACCESS_KEY_ID'],
                        aws_secret_access_key=config['S3_INFO']['SECRET_ACCESS_KEY'],
                        region_name="ap-northeast-2")
#S3에 파일 업로드
def upload_to_s3(file, folder, file_name):
    s3_file_path = f"{folder}{file_name}"
    s3_client.upload_fileobj(file, S3_BUCKET_NAME, s3_file_path)
    
    s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_file_path}"
    print(f"파일 업로드 완료: {s3_url}")
    return s3_url

def generate_tts_for_translation(text, language="Korean"):
    """
    번역된 텍스트에 대해 TTS 생성
    :param text: TTS로 변환할 텍스트
    :param language: 언어 (기본값: Korean)
    :return: 생성된 오디오 파일 경로
    """
    try:
        #OpenAI TTS API 사용
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        #임시 파일로 저장
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file_path = temp_file.name
        temp_file.close()
        
        #오디오 데이터를 파일에 저장
        with open(temp_file_path, "wb") as f:
            f.write(response.content)
        
        return temp_file_path
        
    except Exception as e:
        print(f"TTS 생성 실패: {e}")
        raise e

def create_session_summary(transcripts, main_language="Korean"):
    """
    세션의 모든 대화를 요약하여 한줄 요약과 상세 요약 생성
    :param transcripts: 세션의 모든 대화 기록
    :param main_language: 사용자의 주언어
    :return: 요약 결과 딕셔너리
    """
    #대화 내용을 하나의 텍스트로 결합
    conversation_text = "\n".join([
        f"{t['tag']}: {t['original']}" if t.get('tag') else t['original']
        for t in transcripts
    ])
    
    prompt = f"""
    Create a summary of the following medical conversation in {main_language}.
    
    Conversation:
    {conversation_text}
    
    Please provide:
    1. A one-line summary (maximum 50 characters)
    2. A detailed summary (maximum 300 characters)
    
    Respond in the following JSON format:
    {{
        "one_line_summary": "brief summary",
        "detailed_summary": "detailed summary"
    }}
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        summary_text = response.choices[0].message.content.strip()
        return json.loads(summary_text)
        
    except Exception as e:
        print(f"요약 생성 실패: {e}")
        return {
            "one_line_summary": "대화 요약",
            "detailed_summary": "의료 상담 내용에 대한 요약"
        }