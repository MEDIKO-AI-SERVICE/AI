from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from text_119_utils.stt_tts_translation import transcribe_audio, translate_and_filter_text
from text_119_utils.s3_utils import upload_to_s3, upload_image_to_s3, download_from_s3_image
from text_119_utils.selenium_test import setup_driver
from text_119_utils.en_juso import get_english_address
from text_119_utils.ai_for_form import summarize_content, generate_title_and_type
from text_119_utils.cleaning import clean_form_value
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import os
import time

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#배포 시 도메인 제한하기
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/reportapi/fill_form")
async def fill_form(
    name: str = Form(...),
    number: str = Form(...),
    incident_location: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    emergency_type: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    password: str = Form(...),
    file_1: Optional[UploadFile] = File(None),
    file_2: Optional[UploadFile] = File(None),
    file_3: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    try:
        name = clean_form_value(name)
        phone_number = clean_form_value(number)
        parts = [phone_number[:3], phone_number[3:7], phone_number[7:]]
        location = clean_form_value(incident_location) or clean_form_value(address) or "서울특별시 용산구 청파로47길 100"

        if audio:
            s3_audio_url = upload_to_s3(audio.file, "audio/transcript/")
            transcribed_text, local_audio_path = transcribe_audio(s3_audio_url)
            os.remove(local_audio_path)
            content = transcribed_text

        if not content or content.strip().lower() in ("", "null", "none"):
            content_ko = "신고 내용이 없습니다."
            content_en = "No report content provided."
            processed_content = f"{content_en}({content_ko})"
            default_title_ko = "긴급 신고"
            default_title_en = "Emergency Report"
            default_emergency_type = "Emergency"
        else:
            content_ko, content_en = summarize_content(content)
            processed_content = f"{content_en}({content_ko})"
            default_title_ko, default_title_en, default_emergency_type = generate_title_and_type(processed_content)

        default_title = f"{default_title_en} ({default_title_ko})"[:100]
        emergency_type = clean_form_value(emergency_type) or default_emergency_type
        title = clean_form_value(title) or default_title

        image_urls = []
        for file in [file_1, file_2, file_3]:
            if file:
                image_urls.append(upload_image_to_s3(file.file))

        sido_mapping = {
            "서울특별시": "11", "부산광역시": "26", "대구광역시": "27", "인천광역시": "28",
            "광주광역시": "29", "대전광역시": "30", "울산광역시": "31", "세종특별자치시": "36",
            "경기도": "41", "강원도": "42", "충청북도": "43", "충청남도": "44",
            "전라북도": "45", "전라남도": "46", "경상북도": "47", "경상남도": "48", "제주특별자치도": "49"
        }

        driver = setup_driver()
        driver.get("https://www.119.go.kr/Center119/registEn.do")
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="dsr_name"]')))

        driver.find_element(By.XPATH, '//*[@id="dsr_name"]').send_keys(name)
        driver.find_element(By.XPATH, '//*[@id="call_tel1"]').send_keys(parts[0])
        driver.find_element(By.XPATH, '//*[@id="call_tel2"]').send_keys(parts[1])
        driver.find_element(By.XPATH, '//*[@id="call_tel3"]').send_keys(parts[2])

        select_element = Select(driver.find_element(By.XPATH, '//*[@id="dsrKndCdList"]'))
        for option in select_element.options:
            if option.text.strip().lower() == emergency_type.strip().lower():
                select_element.select_by_visible_text(option.text.strip())
                break

        driver.find_element(By.XPATH, '//*[@id="title"]').send_keys(title)

        for region, code in sido_mapping.items():
            if region in location:
                Select(driver.find_element(By.XPATH, '//*[@id="sidoCode"]')).select_by_value(code)
                time.sleep(1)
                break

        eng_location = get_english_address(location)
        if eng_location:
            address_input = driver.find_element(By.XPATH, '//*[@id="juso"]')
            address_input.clear()
            address_input.send_keys(eng_location)

        driver.find_element(By.XPATH, '//*[@id="contents"]').send_keys(processed_content)
        driver.find_element(By.XPATH, '//*[@id="userPw"]').send_keys(password)

        for idx, s3_url in enumerate(image_urls):
            local_file = download_from_s3_image(s3_url)
            driver.find_element(By.XPATH, f'//*[@id="file_{idx+1}"]').send_keys(local_file)
            os.remove(local_file)

        wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/div/div[2]/div[2]/div/nav/ul/li[2]/button'))).click()

        return {"status": "success", "message": "버튼 클릭 완료"}

    except TimeoutException as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"요소 로딩 시간 초과: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        driver.quit()