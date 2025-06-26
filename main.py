import uuid
from datetime import datetime
from mongodb_utils import get_database
from ai_utils import transcribe_audio, translate_text, summarize_text, detect_language, classify_speaker#,generate_tts
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

@app.post("/api/translation/start")
async def start_session(data: dict = Body(...)):
    start_time = time.time()
    member_id = data.get("member_id")
    if not member_id:
        return JSONResponse(content={"error": "id is required"}, status_code=400)

    session_id = f"session_{uuid.uuid4().hex}"
    main_language = get_main_language(member_id)
    initial_languages = {"Korean", "English"}
    if main_language not in initial_languages and main_language != "Unknown":
        initial_languages.add(main_language)

    sessions_collection.insert_one({
        "_id": session_id,
        "member_id": member_id,
        "transcripts": [],
        "detected_languages": list(initial_languages),
        "created_at": datetime.now(),
        "session_start_time": start_time,
        "ended_at": None
    })

    db["logs"].insert_one({
        "event": "start_session",
        "member_id": member_id,
        "session_id": session_id,
        "timestamp": datetime.now(),
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
    audio_base64 = data.get("audio")
    if not session_id or not audio_base64:
        return JSONResponse(content={"error": "session_id and audio are required"}, status_code=400)

    session = sessions_collection.find_one({"_id": session_id})
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)

    try:
        audio_bytes = base64.b64decode(audio_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        transcript = transcribe_audio(temp_audio_path)
        detected_languages = set(session.get("detected_languages", []))
        translation_result = translate_text(transcript, previous_languages=list(detected_languages))
        translations = translation_result.get("translations", {})

        if "detected_languages" in translation_result:
            detected_languages.update(translation_result["detected_languages"])

        for lang in translation_result.get("detected_languages", []):
            if lang not in translations:
                translations[lang] = {"text": transcript}

        for lang in translations:
            text = translations[lang].get("text", "") if isinstance(translations[lang], dict) else translations[lang]
            translations[lang] = {"text": text} if text.strip() else {}

        tag = classify_speaker(transcript)
        sessions_collection.update_one(
            {"_id": session_id},
            {"$set": {"detected_languages": list(detected_languages)},
             "$push": {"transcripts": {
                 "original": transcript,
                 "translations": translations,
                 "timestamp": datetime.now(),
                 "tag": tag
             }}}
        )

        os.remove(temp_audio_path)
        db["logs"].insert_one({
            "event": "audio_chunk",
            "session_id": session_id,
            "timestamp": datetime.now(),
            "response_time": time.time() - start_time,
            "detected_languages": list(detected_languages),
            "tag": tag
        })

        return JSONResponse(content={
            "message": "Audio processed",
            "session_id": session_id,
            "transcript": transcript,
            "translations": translations,
            "detected_languages": list(detected_languages),
            "tag": tag
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/translation/end")
async def end_session(data: dict = Body(...)):
    session_id = data.get("session_id")
    if not session_id:
        return JSONResponse(content={"error": "session_id is required"}, status_code=400)

    session = sessions_collection.find_one({"_id": session_id})
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)

    session_start_time = session.get("session_start_time", None)
    session_duration = time.time() - session_start_time if session_start_time else None

    db["logs"].insert_one({
        "event": "end_session",
        "session_id": session_id,
        "timestamp": datetime.now(),
        "session_duration": session_duration
    })
    sessions_collection.update_one({"_id": session_id}, {"$set": {"ended_at": datetime.now(), "session_duration": session_duration}})

    return JSONResponse(content={"message": "Session ended", "session_id": session_id})

@app.get("/api/translation/get_languages/{session_id}")
async def get_detected_languages(session_id: str):
    session = sessions_collection.find_one({"_id": session_id})
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)

    transcripts = session.get("transcripts", [])
    first_languages = list(transcripts[0].get("translations", {}).keys()) if transcripts else []

    return JSONResponse(content={"session_id": session_id, "detected_languages": first_languages})

@app.get("/api/translation/scripts/{session_id}/{language}")
async def get_transcripts_by_language(session_id: str, language: str):
    session = sessions_collection.find_one({"_id": session_id})
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)

    transcripts = session.get("transcripts", [])
    first_languages = transcripts[0].get("translations", {}).keys() if transcripts else []
    if language not in first_languages:
        return JSONResponse(content={"session_id": session_id, "language": language, "texts": "", "warning": "Language not available in first transcript"})

    language_texts = []
    for t in transcripts:
        text = t.get("translations", {}).get(language, {}).get("text", "")
        if text:
            tag = t.get("tag", None)
            formatted_text = f"{tag}: {text}" if tag else text
            language_texts.append(formatted_text)

    return JSONResponse(content={"session_id": session_id, "language": language, "texts": language_texts})

@app.get("/api/translation/get_sessions/{member_id}")
async def get_sessions(member_id: str):
    start_time = time.time()
    sessions = sessions_collection.find({"member_id": member_id}, {"_id": 1, "created_at": 1}).sort("created_at", -1)
    session_list = [{"session_id": s["_id"], "created_at": s["created_at"]} for s in sessions]

    if not session_list:
        return JSONResponse(content={"error": "No sessions found for this user"}, status_code=404)

    db["logs"].insert_one({
        "event": "get_sessions",
        "member_id": member_id,
        "timestamp": datetime.now(),
        "response_time": time.time() - start_time,
        "session_count": len(session_list)
    })
    return JSONResponse(content={"member_id": member_id, "sessions": session_list})

@app.get("/api/translation/summary/{session_id}")
async def get_session_summary(session_id: str):
    start_time = time.time()
    session = sessions_collection.find_one({"_id": session_id})
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)

    full_transcript = "\n".join([
        f"{t['tag']}: {t['original']}" if "tag" in t and t["tag"] else t["original"]
        for t in session.get("transcripts", [])
    ])
    summary_text = summarize_text(full_transcript)

    try:
        summary = json.loads(summary_text)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Failed to parse summary"}, status_code=500)

    db["logs"].insert_one({
        "event": "session_summary",
        "session_id": session_id,
        "timestamp": datetime.now(),
        "response_time": time.time() - start_time
    })

    return JSONResponse(content=summary)
