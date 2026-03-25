import google.generativeai as genai


# 이 파일은 오직 '다국어 번역 기능'만 담당합니다.
# 실무에서 가장 필요한 '한<->영 양방향 자동 전환' 및 '기술 용어 보존'에 특화되어 있습니다.
# 기본적으로 고성능인 2.5 Flash를 사용하고, 실패 시 2.0 Flash를 사용합니다.

def translate(text):
    # 함수 이름 변경: translate_email_to_korean -> translate (__init__.py와 통일)

    # 1순위 모델: Gemini 2.5 Flash
    target_model = 'gemini-2.5-flash'

    # 시스템 지침 (페르소나 설정)
    system_instruction = """
    당신은 글로벌 IT 기업의 수석 통역사이자 수석 커뮤니케이션 전문가, 테크니컬 라이터(technical writer)입니다.
    문맥을 파악하여 가장 자연스러운 비즈니스 표현으로 의역(Liberal Translation)해야 하며,
    특히 소프트웨어 엔지니어링 용어와 코드 컨텍스트를 완벽하게 이해하고 보존해야 합니다.
    번역의 품질은 한국 비즈니스 정서에 맞는 '의역'과 '정중함'을 최우선으로 해야 합니다.
    단순한 언어 치환(Translation)을 넘어, 비즈니스 매너에 맞게 문장을 다듬는(Refinement) 역할까지 수행해야 합니다.
    """

    # 프롬프트 구성
    prompt = f"""
    아래 [입력 텍스트]의 언어를 감지하여, 상대 언어(한국어↔영어)로 번역해줘.

    [번역 규칙]
        1. 언어 자동 전환:
           - 입력이 '한국어'라면 -> 원어민이 사용하는 세련된 **Business English**로 의역.
           - 입력이 '영어'라면 -> 한국 비즈니스 정서에 맞는 **정중한 경어체(Business Formal)**를 사용하라.
           - 구어체 필러('음', '그거')나 불필요한 감정 표현은 제거하고, 담백하고 전문적인 어조로 순화하라.

        2. 기술 용어 및 코드 보존 (매우 중요):
           - 코드 블록(`...`), 인라인 코드(`variable`), SQL 쿼리, 로그 메시지는 **절대 번역하지 말고 원문 그대로 유지하라.**
           - 날짜, 시간, 금액, 고유 명사(사람 이름, 회사명)는 절대 변경하거나 누락하지 마라.
           - 고유 명사(AWS, Spark, Kafka)와 업계 표준 용어(Deploy, Commit, Merge, Typos)는 한국어 번역 시에도 영문 또는 통용되는 외래어 표기를 따를 것.
             (예: 'Commits' -> '약속하다(X)', '커밋(O)')

        3. 톤앤매너:
           - (한->영) 직역투를 피하고, 원어민이 쓰는 정중하고 명확한 표현 사용.
           - (영->한) 딱딱한 기계 번역투 대신, "확인 부탁드립니다", "진행 예정입니다" 등 자연스러운 실무 말투 사용.
           - 사람이 직접 타이핑한 것처럼 자연스러운 문단(Paragraph) 구성을 최우선으로 하라.
           - 원문의 문단 구조를 최대한 유지하되, 번역문의 흐름이 끊기지 않도록 자연스러운 연결어를 사용하라.
           - **'AI가 쓴 티가 나는 서식'** (과도한 볼드체, 기계적인 불렛 포인트)은 지양하고, 사람이 쓴 듯한 줄글 형식을 선호하라.

        4. 출력:
           - 부연 설명이나 인사말("여기 번역 결과입니다")을 붙이지 말고, **번역 완료된 텍스트만** 깔끔하게 출력하라.


    [원문 이메일]
    {text}
    """

    try:
        # 모델 설정 (시스템 지침 적용)
        model = genai.GenerativeModel(
            model_name=target_model,
            system_instruction=system_instruction
        )
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        # 1차 시도(2.5) 실패 시 2차 시도(2.0) 폴백 로직
        try:
            print(f"⚠️ 2.5 Flash 번역 실패 ({e}), 2.0 Flash로 전환합니다.")
            fallback_model = 'gemini-flash-latest'
            model = genai.GenerativeModel(fallback_model)
            fallback_prompt = f"다음 텍스트를 정중한 비즈니스 스타일로 번역해줘 (한글↔영어 자동):\n\n{text}"
            response = model.generate_content(fallback_prompt)
            return f"{response.text.strip()} (Note: 2.0 Flash로 수행됨)"
        except Exception as e2:
            print(f"❌ 번역 최종 실패: {e2}")
            return "번역 서비스를 사용할 수 없습니다."