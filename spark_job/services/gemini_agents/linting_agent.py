import google.generativeai as genai


# 이 파일은 오직 '문장 교정/첨삭(Linting)'만 담당합니다.
# 사용자가 대충 쓴 초안을 비즈니스 격식에 맞게 다듬어주는 역할입니다.

def fix_email(text):
    # 함수 이름 변경: generate_draft_reply -> fix_email (__init__.py와 통일)

    target_model = 'gemini-2.0-flash'

    # 시스템 지침 (페르소나 설정)
    system_instruction = """
    당신은 대기업 C-Level 임원 및 주요 클라이언트와 소통하는 '수석 비즈니스 커뮤니케이션 전문가'입니다.
    사용자의 입력은 급하게 작성된 메모나 구어체 등 거친 초안(Draft) 일 수 있습니다. 
    당신의 목표는 이러한 문장을 '비즈니스 이메일' 형태로 완벽하게 교정(Proofreading)하여 
    신뢰감 있고, 명확하며, 격식 있는 형태로 윤문(Polishing)해야 합니다.
    """

    try:
        # 모델 설정
        model = genai.GenerativeModel(
            model_name=target_model,
            system_instruction=system_instruction
        )

        # 프롬프트: '답장'이 아니라 '교정'을 요청하는 내용으로 변경
        prompt = f"""
        아래 [사용자 초안]을 비즈니스 메일 격식에 맞게 자연스럽게 수정(Fix/Linting)해줘.

        [수정 규칙]
        1. 톤앤매너(Tone & Manner) : 오탈자를 교정하고, 구어체나 가벼운 표현을 정중한 문체(Business Formal)로 다듬을 것.
            - 상대방을 존중하되, 비굴하지 않고 세련된 전문적인 어조를 유지하라.
            
        2. 내용 유지(Fact Keeping) : 원래 의미나 의도가 왜곡되지 않도록 주의할 것.
            - 문장 흐름을 매끄럽게 정리하되, 불필요하게 내용을 덧붙이지 말 것.
            - 날짜, 시간, 금액, 사람 이름, 고유 명사는 절대 누락하거나 변경하지 말 것.
            
        3. 구조화(Structuring) :
           - 핵심 용건을 문두에 배치하라(두괄식).
           - 3가지 이상의 나열된 정보(날짜, 항목, 요청사항 등)가 있다면, 줄글 대신 글머리 기호(Bullet points)를 사용하여 가독성을 높여라.
    
        4. 내용 정제(Refinement) :
           - '음...', '그거', '저기' 같은 모호한 지시어나 구어체 필러(Filler)를 제거하라.
           - 부정적인 감정(화남, 짜증)이 섞인 표현은 '객관적인 상황 묘사'나 '정중한 요청'으로 순화하라.
        
        5. 자연스러운 줄글 지향 : **굵게 표시(**...**)**, 제목(#), 과도한 글머리 기호 등 'AI가 쓴 티가 나는 서식'을 절대 사용하지 마라.
            - 사람이 직접 타이핑한 것처럼 자연스러운 문단(Paragraph) 구성을 최우선으로 하라.
            - 항목이 3개 이하일 경우 굳이 불렛 포인트를 쓰지 말고, 문장 내에서 자연스럽게 연결하라.

        6. 출력 형식 : 결과물은 수정된 본문 텍스트만 출력할 것. (설명 제외)
            - '안녕하세요'로 시작해서 '감사합니다'로 끝나는 이메일 본문 텍스트만 출력하라. (마크다운 서식 없음)

        [사용자 초안]
        {text}
        """

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        # 폴백: 1.5 Flash 사용
        try:
            fallback_model = 'gemini-flash-latest'
            model = genai.GenerativeModel(fallback_model)

            fallback_prompt = f"다음 내용을 비즈니스 메일 격식에 맞춰 정중하고 자연스럽게 교정해줘 (결과만 출력):\n\n{text}"

            response = model.generate_content(fallback_prompt)
            # 데모용이므로 폴백 여부를 티가 나게 남겨두거나, 원하시면 지우셔도 됩니다.
            return f"{response.text.strip()} "
        except Exception as e2:
            print(f"❌ 문장 교정 실패: {e2}")
            return "죄송합니다. 문장을 교정하는 중 오류가 발생했습니다."