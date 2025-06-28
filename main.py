import uuid
from datetime import datetime
from mongodb_utils import get_database
from ai_utils import download_audio_from_s3_presigned_url, transcribe_audio, translate_text_simple, detect_language_simple, upload_to_s3, generate_tts_for_translation, create_session_summary
import base64
import tempfile
import os
from main_language import get_main_language
import time
import json
from fastapi import FastAPI, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pytz import timezone

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#프론트 배포 후 도메인 기재
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = get_database()
sessions_collection = db["sessions"]
#MongoDB 인덱스 추가(쿼리 성능 고려)
sessions_collection.create_index([("created_at", -1)])

@app.post("/api/translation/start/{member_id}")
async def start_session(member_id: int):
    start_time = time.time()
    session_id = f"session_{uuid.uuid4().hex}"
    main_language = get_main_language(member_id)
    print('main language:', main_language)
    #초기 언어 설정: Korean과 main_language만 추가
    initial_languages = {"Korean"}
    if main_language and main_language != "Unknown":
        initial_languages.add(main_language)
    else:
        #main_language가 없거나 Unknown인 경우에만 English 추가
        initial_languages.add("English")
    print('initial languages:', initial_languages)
    sessions_collection.insert_one({
        "_id": session_id,
        "member_id": member_id,
        "transcripts": [],
        "detected_languages": list(initial_languages),
        "main_language": main_language,
        "created_at": str(datetime.now()),
        "session_start_time": start_time,
        "ended_at": None
    })
    db["logs"].insert_one({
        "event": "start_session",
        "member_id": member_id,
        "session_id": session_id,
        "timestamp": str(datetime.now()),
        "response_time": time.time() - start_time
    })
    return JSONResponse(content={
        "message": "Session started",
        "session_id": session_id,
        "main_language": main_language,
        "detected_languages": list(initial_languages)
    }, status_code=201)

@app.post("/api/translation/chat")
async def handle_audio_chunk(data: dict = Body(...)):
    start_time = time.time()
    session_id = data.get("session_id")
    text = data.get("text")
    audio_presigned_url = data.get("audio")
    tag = data.get("tag")
    #session_id는 필수
    if not session_id:
        return JSONResponse(content={"error": "session_id is required"}, status_code=400)
    #text와 audio 중 하나는 반드시 있어야 함
    if not text and not audio_presigned_url:
        return JSONResponse(content={"error": "Either text or audio is required"}, status_code=400)
    #tag는 필수 (0: 환자, 1: 의료진)
    if tag not in [0, 1]:
        return JSONResponse(content={"error": "tag is required and must be 0 (patient) or 1 (medical staff)"}, status_code=400)
    session = sessions_collection.find_one({"_id": session_id})
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)
    try:
        transcript = ""
        temp_audio_path = None
        #오디오 처리
        if audio_presigned_url:
            temp_audio_path = download_audio_from_s3_presigned_url(audio_presigned_url)
            audio_transcript = transcribe_audio(temp_audio_path)
            transcript = audio_transcript
        #텍스트 처리 (오디오와 텍스트가 모두 있으면 합침)
        if text:
            if transcript:
                transcript = f"{transcript} {text}"
            else:
                transcript = text
        #언어 감지 및 번역 처리 (단순화된 버전)
        detected_language = detect_language_simple(transcript)
        main_language = session.get("main_language", "English")
        #단순화된 번역: 0(환자)→한국어, 1(의료진)→main_language
        target_tag = 0 if tag == 0 else 1
        translation_result = translate_text_simple(transcript, target_tag, main_language)
        translations = translation_result.get("translations", {})
        #감지된 언어를 세션에 추가 (새로운 언어인 경우)
        detected_languages = set(session.get("detected_languages", []))
        if detected_language and detected_language != "Unknown":
            detected_languages.add(detected_language)
        #TTS 생성 및 S3 업로드
        tts_url = None
        target_language = "Korean" if tag == 0 else main_language
        if target_language in translations and translations[target_language].get("text"):
            translated_text = translations[target_language]["text"]
            try:
                tts_path = generate_tts_for_translation(translated_text, target_language)
                with open(tts_path, "rb") as audio_file:
                    tts_url = upload_to_s3(audio_file, f"tts/{session_id}/", f"chat_{len(session.get('transcripts', [])) + 1}_{target_language}.mp3")
                os.remove(tts_path)
            except Exception as e:
                print(f"TTS 생성 실패: {e}")
        #세션 업데이트
        transcript_obj = {
            "chat_id": len(session.get("transcripts", [])) + 1,  #순서 ID 추가
            "original": transcript,
            "translations": translations,
            "timestamp": str(datetime.now()),
            "tag": tag,
            "detected_language": detected_language,
            "tts": tts_url
        }
        sessions_collection.update_one(
            {"_id": session_id},
            {"$set": {"detected_languages": list(detected_languages)},
             "$push": {"transcripts": transcript_obj}}
        )
        #임시 파일 정리
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        #로그 기록
        db["logs"].insert_one({
            "event": "chat",
            "session_id": session_id,
            "timestamp": str(datetime.now()),
            "response_time": time.time() - start_time,
            "detected_languages": list(detected_languages),
            "tag": tag,
            "has_audio": bool(audio_presigned_url),
            "has_text": bool(text),
            "detected_language": detected_language
        })
        return JSONResponse(content={
            "message": "Audio/Text processed",
            "session_id": session_id,
            "transcript": transcript,
            "translations": translations,
            "detected_languages": list(detected_languages),
            "tag": tag,
            "detected_language": detected_language,
            "tts": tts_url
        })
    except Exception as e:
        #임시 파일 정리 (에러 발생 시에도)
        if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/translation/end/{session_id}")
async def end_session(session_id: str):
    start_time = time.time()
    session = sessions_collection.find_one({"_id": session_id})
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)
    transcripts = session.get("transcripts", [])
    if not transcripts:
        #대화가 없는 빈 세션은 삭제
        sessions_collection.delete_one({"_id": session_id})
        db["logs"].insert_one({
            "event": "delete_empty_session",
            "session_id": session_id,
            "timestamp": str(datetime.now()),
            "response_time": time.time() - start_time
        })
        return JSONResponse(content={
            "message": "Empty session deleted",
            "session_id": session_id
        })
    try:
        session_start_time = session.get("session_start_time", None)
        session_duration = time.time() - session_start_time if session_start_time else None
        main_language = session.get("main_language", "Korean")
        #1. 세션 요약문 생성
        summary_result = create_session_summary(transcripts, main_language)
        #2. 세션 종료 및 데이터 저장
        sessions_collection.update_one(
            {"_id": session_id}, 
            {
                "$set": {
                    "ended_at": str(datetime.now()),
                    "session_duration": session_duration,
                    "summary": summary_result,
                    "transcripts": transcripts
                }
            }
        )
        db["logs"].insert_one({
            "event": "end_session",
            "session_id": session_id,
            "timestamp": str(datetime.now()),
            "session_duration": session_duration,
            "response_time": time.time() - start_time
        })
        return JSONResponse(content={
            "message": "Session ended successfully",
            "session_id": session_id,
            "session_duration": session_duration
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#세션 목록 조회
@app.get("/api/translation/session_list/{member_id}")
async def get_session_list(member_id: int):
    sessions = sessions_collection.find({
        "member_id": member_id,
        "transcripts.0": {"$exists": True}
    }, {"_id": 1, "created_at": 1, "summary": 1}).sort("created_at", -1)
    session_list = []
    for s in sessions:
        session_id = s["_id"]
        one_line_summary = s.get("summary", {}).get("one_line_summary", "")
        created_at = s.get("created_at")
        if isinstance(created_at, datetime):
            created_at = created_at.strftime("%Y-%m-%d")
        elif isinstance(created_at, str):
            created_at = created_at[:10]
        else:
            created_at = ""
        session_list.append({
            "session_id": session_id,
            "one_line_summary": one_line_summary,
            "created_at": created_at
        })
    return JSONResponse(content={"member_id": member_id, "sessions": session_list})

#세션 상세 조회
@app.get("/api/translation/session_detail/{member_id}/{session_id}")
async def get_session_detail(member_id: int, session_id: str):
    session = sessions_collection.find_one({"_id": session_id, "member_id": member_id})
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)
    summary = session.get("summary", {})
    transcripts = session.get("transcripts", [])
    scripts = []
    for t in transcripts:
        original = t.get('original', '')
        translations = {}
        for lang, val in t.get("translations", {}).items():
            text = val.get("text") if isinstance(val, dict) else val
            if text:
                translations[lang] = text
        script_item = {
            "chat_id": t.get("chat_id"),
            "original": original,
            "translations": translations,
            "tag": t.get("tag"),
            "tts": t.get("tts")
        }
        scripts.append(script_item)
    created_at = session.get("created_at")
    if isinstance(created_at, datetime):
        created_at = created_at.strftime("%Y-%m-%d")
    elif isinstance(created_at, str):
        created_at = created_at[:10]
    else:
        created_at = ""
    return JSONResponse(content={
        "session_id": session_id,
        "created_at": created_at,
        "one_line_summary": summary.get("one_line_summary", ""),
        "detailed_summary": summary.get("detailed_summary", ""),
        "scripts": scripts
    })
