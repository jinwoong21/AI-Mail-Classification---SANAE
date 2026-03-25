import google.generativeai as genai

# 이 파일은 오직 '할 일(Action Item) 추출 기능'만 담당합니다.
# API 키 설정은 상위(__init__.py)에서 이미 완료된 상태로 넘어옵니다.

def extract_todos(text):
    target_model = 'gemini-2.0-flash'

    # 시스템 지침 (페르소나 설정 : 단순 추출기가 아니라 'GTD(Getting Things Done) 전문가')
    system_instruction = """
    당신은 업무 효율을 극대화하는 'GTD(Getting Things Done) 태스크 매니저'입니다.
    사용자의 이메일에서 실행 가능한(Actionable) 작업만을 추출하여, 
    즉시 체크리스트로 등록할 수 있는 형태로 변환해야 합니다.
    """

    try:
        # 모델 설정 (이미 설정된 api_key 사용)
        model = genai.GenerativeModel(
            model_name=target_model,
            system_instruction=system_instruction
        )

        # 프롬프트 구성
        prompt = f"""
        아래 [이메일 본문]을 분석하여 '할 일 목록(To-Do List)'을 작성해줘.

        [추출 규칙]
        
        1. 형식: `- 동작+목적어` 형태의 체크리스트로 출력할 것. (예: 보고서 검토하기)
        
        2. 메타데이터 표시 (이모지 활용):
           - 마감기한(Deadline), 담당자, 특정 시간 등 상세 정보가 언급되면 볼드체로 표시.
           - 마감기한이 있다면 끝에 `📅 MM/DD` 추가.
           - 긴급한 내용("급함", "ASAP" 등)이라면 앞에 `🔥` 추가.
           - 특정 담당자가 지정되었다면 끝에 `👤 담당자명` 추가.
           
        3. 내용 정제:
           - "확인 부탁드립니다" 같은 서술형을 "확인하기" 같은 **명령/행동형**으로 변환할 것.
           - 모호한 내용은 구체화할 것 (예: "그거 보내줘" -> "회의록 파일 송부하기")
           - 명확한 지시사항이나 요청사항만 추출할 것. (단순 인사치레, 정보 공유는 제외)
           
        4. 예외 처리:
           - 수행할 작업이 없는 단순 공지나 스팸 메일 이라면, "✅ 특이사항 없음 (단순 참조/공지)" 라고만 출력할 것.
           
        5. 결과물 외의 다른 설명은 절대 추가하지 말 것.
        
        6. 언어 :
           - 원문 언어와 상관없이 요약 결과는 무조건 **한국어**로 작성할 것.

        [이메일 본문]
        {text}
        """

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        # 1차 시도 실패 시 폴백 (Fallback) 로직: 1.5 Flash 사용
        try:
            fallback_model = 'gemini-flash-latest'
            model = genai.GenerativeModel(fallback_model)
            fallback_prompt = f"다음 메일에서 할 일만 뽑아서 '- [ ] 할일' 형식으로 써줘:\n\n{text}"

            response = model.generate_content(fallback_prompt)
            return f"{response.text.strip()}"
        except Exception as e2:
            print(f"❌ 할 일 추출 에러 발생: {e2}")
            return "할 일 목록을 추출할 수 없습니다."