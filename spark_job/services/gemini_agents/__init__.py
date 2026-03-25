import google.generativeai as genai
import os
from .summary_agent import summarize
from .todo_agent import extract_todos
from .replygen_agent import generate_reply
from .linting_agent import fix_email
from .translate_agent import translate


class GeminiAgentService:
    """
    Gemini 관련 모든 기능을 하나로 통합하여 제공하는 서비스 클래스
    """
    # [수정 1] 여기에 적어두신 키를 바로 사용하겠습니다.
    DEFAULT_API_KEY = 'AIza...'

    def __init__(self, key_path=None):
        # [수정 2] 복잡한 파일 읽기 로직을 제거하고 바로 키를 설정합니다.
        try:
            # 파일 경로가 들어와도 무시하고, 위에서 설정한 키를 우선 사용하거나
            # 파일 읽기에 실패하면 하드코딩된 키를 사용하는 방식으로 변경
            api_key = self.DEFAULT_API_KEY

            # (선택사항) 만약 꼭 파일에서 읽어야 한다면 주석을 풀어서 사용하세요
            # if key_path and os.path.exists(key_path):
            #     with open(key_path, 'r', encoding='utf-8') as f:
            #         api_key = f.read().strip()

            if not api_key:
                raise ValueError("API Key is empty")

            genai.configure(api_key=api_key)
            print("✨ Gemini Agent Service Initialized (API Configured)")
            self.ready = True

        except Exception as e:
            print(f"❌ Gemini Init Error: {e}. AI 기능이 비활성화됩니다.")
            self.ready = False

    # ----------------------------------------------------
    # 각 Agent 파일의 함수를 이 클래스 메서드로 연결 (Wrapping)
    # ----------------------------------------------------
    def summarize(self, text):
        if not self.ready: return "AI 서비스 비활성화"
        return summarize(text)

    def extract_todos(self, text):
        if not self.ready: return "AI 서비스 비활성화"
        return extract_todos(text)

    def generate_reply(self, text, intent):
        if not self.ready: return "AI 서비스 비활성화"
        return generate_reply(text, intent)

    def fix_email(self, text):
        if not self.ready: return "AI 서비스 비활성화"
        return fix_email(text)

    def translate(self, text):
        if not self.ready: return "AI 서비스 비활성화"
        return translate(text)


# [수정 3] 경로가 틀려도 상관없도록 수정
def get_gemini_service(base_dir):
    # 파일 경로를 넘기긴 하지만, 위에서 하드코딩된 키를 쓰므로 에러가 나지 않습니다.
    # 경로는 더미(dummy)로 넘겨도 됩니다.
    return GeminiAgentService(None)