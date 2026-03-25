import google.generativeai as genai


# 이 파일은 오직 '답장 초안 생성'만 담당합니다.
# API 키 설정은 상위(__init__.py)에서 이미 완료된 상태로 넘어옵니다.

def generate_reply(text, intent="확인 후 연락 바람"):
    # 함수 이름 변경: generate_draft_reply -> generate_reply (__init__.py와 통일)

    target_model = 'gemini-2.0-flash'

    # 🌟 System Instruction (페르소나 설정)
    system_instruction = """
    당신은 일 잘하는 'AI 비서'이자 '비즈니스 커뮤니케이션 전문가'입니다. 
    사용자의 의도를 정확히 파악하여, 격식 있고 정중한 하되,
    실제 직장인이 작성한 것처럼 담백하고 명확한 비즈니스 이메일 답장을 작성해야합니다.
    """

    try:
        # 모델 설정 (시스템 지침 적용)
        model = genai.GenerativeModel(
            model_name=target_model,
            system_instruction=system_instruction
        )

        prompt = f"""
        아래 [받은 메일]을 읽고, [답장 방향]을 반영하여 최적의 답장 초안을 작성해줘.

        [작성 규칙]
        1. 사람이 쓴 듯한 서식 : **굵게 표시(**...**)**, 제목 태그(#) 등 마크다운 서식을 절대 사용하지 마라.
           - 자연스러운 줄바꿈과 문단 구성을 사용하라.
           - [답장 방향]의 의도가 명확히 드러나되, 비즈니스 매너를 갖춘 공손한 말투(Business Formal)를 사용할 것.
           
        2. 정보 활용 및 빈칸 처리 :
           - [받은 메일]에 있는 정보(날짜, 시간, 안건 등)는 최대한 활용하여 답장에 녹여내라.
           - 내가 채워넣어야 할 미확정 정보(구체적 일정, 장소 등)가 있다면 임의로 지어내지 말고 `[ ]` 괄호로 표시해라. (예: `[날짜]`, `[시간]`)
        
        3. 톤앤매너 (Intent 반영) :
           - [답장 방향]이 '거절'이나 '지연'이라면, 쿠션어(송구합니다만, 양해 부탁드립니다)를 적절히 사용하라.
           - [답장 방향]이 '수락'이나 '진행'이라면, 군더더기 없이 명확하게 의사를 밝혀라.
           - 정중하되, 너무 딱딱한 문어체보다는 자연스러운 비즈니스 메일 말투를 사용.
           - [답장 방향]을 충실히 반영하되, 거절 시에는 완곡하게 표현할 것.

        4. 형식 :
           - **제목(Subject) 라인은 작성하지 마.** (본문만 출력)
           - 첫 인사는 "OOO님, 안녕하세요." 또는 "안녕하세요, OOO님." 형태로 시작할 것.
           - 마지막에는 "감사합니다. [내 이름] 드림" 형태로 끝인사를 하며 마칠 것.
           - 상대방의 이름이나 직급이 있으면 적절한 호칭(님, 팀장님 등)으로 정중하게 시작하고, 끝인사(감사합니다 등)로 마무리하라.
           - 불필요한 미사여구 및 서두("네, 알겠습니다" 등) 없이 바로 본론으로 들어가라.
           - 구체적인 날짜나 장소가 정해지지 않았다면 "[ ]"와 같이 괄호로 비워 사용자가 채우게 할 것.
           - 결과물 외에 다른 설명은 덧붙이지 말 것.
        
        [받은 메일]
        {text}

        [답장 방향 (Intent)]
        {intent}
        """

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        # 1차 시도 실패 시 1.5 Flash로 폴백 (시스템 지침 없이 프롬프트에 의존)
        try:
            fallback_model = 'gemini-flash-latest'
            model = genai.GenerativeModel(fallback_model)

            # 1.5 모델용 프롬프트 (페르소나 직접 주입)
            fallback_prompt = f"너는 비즈니스 AI 비서야. 다음 메일에 대해 '{intent}'라는 의도로 정중하게 답장을 써줘:\n\n{text}"

            response = model.generate_content(fallback_prompt)
            return f"{response.text.strip()} (Note: 1.5 Flash로 작성됨)"
        except Exception as e2:
            print(f"❌ 답장 생성 실패: {e2}")
            return "답장 초안을 생성할 수 없습니다."