import google.generativeai as genai

# 이 파일은 오직 '요약 기능'만 담당합니다.
# API 키 설정은 상위(__init__.py)에서 이미 완료된 상태로 넘어옵니다.

def summarize(text):
    target_model = 'gemini-2.0-flash'

    # 시스템 지침 (페르소나 설정)
    system_instruction = """
    당신은 바쁜 경영진을 위해 이메일의 핵심을 1초 만에 파악할 수 있도록 돕는 'Executive Summary Bot'입니다.
    사용자에게 핵심 정보를 빠르게 전달하는 것을 최우선으로 하며, 불필요한 미사여구는 모두 제거하고, 
    사실(Fact)과 할 일(Action) 위주로 정보를 구조화하십시오.
    단순 요약을 넘어, 숨겨진 정보(소요 시간, 중요도 등)를 계산하여 비즈니스 인사이트를 제공하십시오.
    아래 요약 규칙을 엄격하게 준수해야 합니다. 
    출력은 요약 결과만 포함해야 합니다.
    """

    try:
        # 모델 설정 (이미 설정된 api_key 사용)
        model = genai.GenerativeModel(
            model_name=target_model,
            system_instruction=system_instruction
        )

        # 프롬프트 구성
        prompt = f"""
        아래 [메일 원문]을 분석하여 다음 규칙에 따라 요약 보고서를 작성해.

        [규칙]
        1. 광고/스팸 판단 :
           - 홍보, 뉴스레터, 단순 광고성 메일이라면 상세 요약 없이 "📢 [광고] 프로모션/뉴스레터 메일입니다." 라고만 출력하고 종료할 것.
           
        2. 언어 :
           - 원문 언어와 상관없이 요약 결과는 무조건 **한국어**로 작성할 것.

        3. 출력 구조 (Markdown) :
           - 💡 한 줄 요약:** 전체 내용을 한 문장으로 압축.
           - 📝 핵심 내용:** 주요 사실(날짜, 금액, 이슈 등)을 글머리 기호(Bullet points)로 3개 이내 정리. (명사형 종결: ~함, ~임)
           - ✅ 요청/할 일:** 수신자가 취해야 할 행동이 있다면 명시. 없으면 "특이사항 없음" 출력.
        
        4. 데이터 정밀화 (중요) :
           - **시간 계산:** 시작 시간과 종료 시간이 있다면, **자동으로 소요 시간을 계산하여 괄호 안에 병기할 것.** (예: 14:00~16:00 -> 14:00~16:00 (2시간))
           - **타임존:** KST, EST 등 타임존 정보가 있다면 누락하지 말고 반드시 표기할 것.
           - 금액/수치: 원문의 숫자를 정확히 유지할 것.

        5. 주의사항 :
           - 날짜, 시간, 장소, 금액은 원문의 수치를 정확히 유지하며 일정/회의는 날짜와 시간을 강조하여 명시.
        

        [메일 원문]
        {text}
        """

        # 실행
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        # 에러 발생 시 폴백 (1.5 Flash)
        try:
            fallback_model = 'gemini-flash-latest'
            model = genai.GenerativeModel(fallback_model)
            fallback_prompt = f"다음 이메일을 한국어로 짧고 굵게 3줄 요약해줘:\n\n{text}"

            response = model.generate_content(fallback_prompt)
            return f"{response.text.strip()} (Note: 요약 모델 Fallback 동작)"
        except Exception as e2:
            print(f"⚠️ 요약 에러 발생: {e2}")
            return "요약 정보를 가져올 수 없습니다."