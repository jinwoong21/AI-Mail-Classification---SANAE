import google.generativeai as genai
import pandas as pd
import os
import time
import ast
import re
import random

# ==========================================
# [설정] 니 API 키를 따옴표 안에 넣어라.
API_KEY = "YOUR_API_KEY_HERE"
# ==========================================

# 파일 설정 (니가 요청한 대로 수정함)
PROMPT_FILE = "update_data_prompt.txt"  # 새로운 규칙 파일
OUTPUT_FILE = "data/final_result_2.csv"  # 기존 파일에 이어서 저장함
TOTAL_COUNT = 3000  # 추가 생성 3000개
BATCH_SIZE = 10  # 10개씩 끊어서 요청
MAX_RETRIES = 10  # 429 뜨면 최대 10번 재시도

if API_KEY == "YOUR_API_KEY_HERE":
    print("❌ 야, 키 안 넣었잖아. 코드 열어서 API_KEY부터 채워.")
    exit()

# Gemini 2.0 Flash 설정
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')


def load_prompt(filepath):
    """규칙 파일 읽어오기"""
    if not os.path.exists(filepath):
        print(f"❌ 야, '{filepath}' 파일이 없잖아. 같은 폴더에 둬야지.")
        exit()
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def generate_batch_data(base_prompt, start_id, count):
    """Gemini에게 프롬프트 던져서 데이터 받아오기 (429 방어 로직 포함)"""

    final_instruction = f"""
    {base_prompt}

    [SYSTEM INSTRUCTION]
    위의 '추가 생성 규칙'을 완벽하게 준수하여, 이번에 총 {count}개의 이메일 데이터를 생성하시오.

    1. email_id는 {start_id}부터 {start_id + count - 1}까지 순차적으로 부여하시오.
    2. 출력 포맷은 반드시 Python List 형식인 `info = [...]` 형태로만 출력하시오.
    3. 코드 블록(```python)이나 설명, 사족은 절대 붙이지 말고 오직 데이터 리스트만 출력하시오.
    4. received_at은 datetime 객체 말고 'YYYY-MM-DD HH:MM:SS' 형식의 문자열로 출력하시오.
    """

    # 지수 백오프 (Exponential Backoff) 로직
    for attempt in range(MAX_RETRIES):
        try:
            # API 호출
            response = model.generate_content(final_instruction)
            return response.text.strip()

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Resource exhausted" in error_msg:
                wait_time = min(60, (2 ** attempt) * 2 + random.uniform(0, 1))
                print(f"   ⚠️ [429 속도제한] 구글 형님이 화났다. {wait_time:.1f}초 쉬고 다시 빈다... (시도 {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                print(f"   ⚠️ 알 수 없는 에러: {e}. 5초 쉼.")
                time.sleep(5)

    print("   ❌ 10번이나 빌었는데 안 받아줌. 이 구간은 포기하고 넘어간다.")
    return ""


def parse_llm_response(text):
    """LLM이 뱉은 문자열을 파이썬 리스트로 변환"""
    data_list = []
    try:
        # 마크다운 제거
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()

        # info = [...] 패턴 찾기
        match = re.search(r"info\s*=\s*(\[.*\])", text, re.DOTALL)

        list_str = ""
        if match:
            list_str = match.group(1)
        elif text.startswith("[") and text.endswith("]"):
            list_str = text

        if list_str:
            data_list = ast.literal_eval(list_str)

    except Exception as e:
        print(f"⚠️ 파싱 실패 (AI가 헛소리함): {e}")

    return data_list


# --- 메인 실행부 ---
print(f"=== Gemini 추가 데이터 생성기 (목표: {TOTAL_COUNT}개 추가) ===")
print(f"📄 규칙 파일: {PROMPT_FILE}")

# 1. 프롬프트 로드
base_prompt_content = load_prompt(PROMPT_FILE)

current_id = 10001
generated_total = 0

# 2. 이어쓰기 로직 (파일 있으면 ID 확인해서 그 뒤부터 시작)
if os.path.exists(OUTPUT_FILE):
    try:
        existing_df = pd.read_csv(OUTPUT_FILE)
        if not existing_df.empty:
            last_id = existing_df['email_id'].max()
            # 기존 개수와 상관없이, 이번 실행에서 3000개를 '추가'로 만드는 거니까
            # ID만 이어서 설정하고, generated_total은 0부터 시작해서 3000 채울 때까지 돌림.
            current_id = int(last_id) + 1
            print(f"🔄 기존 파일 발견! 마지막 ID: {last_id}")
            print(f"🚀 ID {current_id}부터 시작해서 {TOTAL_COUNT}개 더 만든다.")
    except:
        print("⚠️ 파일 읽기 실패. 그냥 덮어쓰거나 새로 만듦.")

# 3. 배치 루프
while generated_total < TOTAL_COUNT:
    # 남은 개수 계산
    current_batch_size = min(BATCH_SIZE, TOTAL_COUNT - generated_total)

    print(f"🚀 추가 배치 생성 중... (ID: {current_id} ~ {current_id + current_batch_size - 1})")

    # API 호출
    raw_response = generate_batch_data(base_prompt_content, current_id, current_batch_size)

    # 파싱
    batch_data = parse_llm_response(raw_response)

    if batch_data and len(batch_data) == current_batch_size:
        # 성공 시 저장
        df = pd.DataFrame(batch_data)

        # 파일 헤더 처리 (파일이 없으면 헤더 쓰고, 있으면 뺌)
        header_mode = not os.path.exists(OUTPUT_FILE)
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='a', header=header_mode)

        generated_total += len(batch_data)
        current_id += len(batch_data)

        print(f"   ✅ 저장 완료! (이번 실행 누적: {generated_total}/{TOTAL_COUNT})")

        time.sleep(1)

    else:
        print("   💩 파싱 실패하거나 개수 안 맞음. 이 구간 건너뜀.")
        # 실패 시 ID 점프
        current_id += current_batch_size
        generated_total += current_batch_size

print(f"\n🎉 작업 끝. 결과 파일: {OUTPUT_FILE}")