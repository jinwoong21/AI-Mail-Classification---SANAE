import os
import multiprocessing
from .gemini_agents import get_gemini_service

DEPT_MAP = {0: "개발팀", 1: "서비스팀", 2: "영업팀", 3: "마케팅팀", 4: "경영팀", 5: "인사팀", 6: "기획팀", 7: "운영팀", 8: "재무팀", 9: "보안팀", 10:"디자인팀"}
ASSIGNEE_MAP = {0: "실무 담당자", 1: "관리자 급", 2: "책임자 급"}


def run_spam_check(data, model_path, token_path, scaler_path, return_dict):
    """ 스팸 분석 전용 프로세스 """
    try:
        from spark_job.services.spam_service import SpamService

        # [수정] scaler_path (jh/scalers.pkl) 전달
        service = SpamService(model_path, token_path, scaler_path)

        # [수정] data 전체 전달 (is_spam 내부에서 4가지 입력 추출)
        is_spam, spam_conf = service.is_spam(data)

        return_dict['result'] = (is_spam, spam_conf)
    except Exception as e:
        return_dict['error'] = str(e)
        print(f"❌ Spam Process Error: {e}")


def run_main_analysis(data, model_path, scaler_path, return_dict):
    """ 메인 분류 전용 프로세스 """
    try:
        from spark_job.services.main_service import MainClassifierService

        # MainService는 ml/scaler.pkl 사용
        service = MainClassifierService(model_path, scaler_path, device="cpu")
        raw_res = service.predict(data)

        return_dict['result'] = raw_res
    except Exception as e:
        return_dict['error'] = str(e)
        print(f"❌ Main Process Error: {e}")


class EmailAgent:
    def __init__(self, base_dir="/flask_web"):
        print(f"🕵️ Agent Initialized (Multi-processing Mode). Base Dir: {base_dir}")
        self.base_dir = base_dir

        # =========================================================
        # [경로 설정] 정확한 폴더 구조 반영 (jh vs ml)
        # =========================================================
        # 1. Spam Model (LSTM) -> jh 폴더
        self.spam_model_path = os.path.join(base_dir, "jh", "best_model_lstm.h5")
        self.spam_token_path = os.path.join(base_dir, "jh", "tokenizers.pkl")
        self.spam_scaler_path = os.path.join(base_dir, "jh", "scalers.pkl")  # 여기에 lstm용 스케일러 포함됨

        # 2. Main Model (Bert) -> ml 폴더
        self.main_model_path = os.path.join(base_dir, "ml", "best_model3.pt")
        self.main_scaler_path = os.path.join(base_dir, "ml", "scaler.pkl")  # 여기는 단독 스케일러

        # Gemini Init
        current_service_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"🤖 Initializing Gemini Agent Service from: {current_service_dir}")
        self.gemini_service = get_gemini_service(current_service_dir)

    def process_email(self, data):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        # =========================================================
        # [Step 1] 스팸 체크
        # =========================================================
        print("🏃 [Step 1] Spawning Spam Process...")
        p1 = multiprocessing.Process(
            target=run_spam_check,
            # jh 폴더에 있는 파일들 전달
            args=(data, self.spam_model_path, self.spam_token_path, self.spam_scaler_path, return_dict)
        )
        p1.start()
        p1.join()

        if 'error' in return_dict:
            return {"status": "ERROR", "message": f"Spam Check Failed: {return_dict['error']}"}

        is_spam, spam_conf = return_dict.get('result', (False, 0.0))

        if is_spam:
            print("   👉 Detected as SPAM.")
            return {
                "status": "BLOCKED",
                "type": "SPAM",
                "confidence": spam_conf,
                "ui_data": {"folder": "스팸메일함", "tags": ["🚫 스팸"]}
            }

        # =========================================================
        # [Step 2] 상세 분석
        # =========================================================
        print("🏃 [Step 2] Spawning Main Analysis Process...")
        return_dict = manager.dict()

        p2 = multiprocessing.Process(
            target=run_main_analysis,
            # ml 폴더에 있는 파일들 전달
            args=(data, self.main_model_path, self.main_scaler_path, return_dict)
        )
        p2.start()
        p2.join()

        if 'error' in return_dict:
            return {"status": "ERROR", "message": f"Main Analysis Failed: {return_dict['error']}"}

        raw_res = return_dict.get('result')
        if raw_res is None:
            return {"status": "ERROR", "message": "Main Analysis returned None"}

        # =========================================================
        # [Step 3] 결과 정리
        # =========================================================
        response = {
            "status": "SUCCESS",
            "type": "NORMAL",
            "raw_result": raw_res,
            "ui_data": {
                "tags": [],
                "folder": "받은편지함",
                "assignee_suggestion": None,
                "ai_summary": "",
                "ai_todo": "",
                "ai_reply": ""
            }
        }

        try:
            dept_id = raw_res.get('label_dept', {}).get('class', 9)
            dept_name = DEPT_MAP.get(dept_id, "기타")
            response['ui_data']['folder'] = dept_name

            if raw_res.get('is_complaint', {}).get('class') == 1:
                response['ui_data']['tags'].append("🚨 민원")

            p_class = raw_res.get('priority_level', {}).get('class', 0)
            p_conf = raw_res.get('priority_level', {}).get('confidence', 0.0)
            if p_class >= 1:
                if p_class == 2 and p_conf > 0.8:
                    response['ui_data']['tags'].append("🔥 매우 긴급")
                else:
                    response['ui_data']['tags'].append("⚡ 긴급")

            a_id = raw_res.get('assignee', {}).get('class', 0)
            a_name = ASSIGNEE_MAP.get(a_id, "미정")
            a_conf = raw_res.get('assignee', {}).get('confidence', 0.0)
            response['ui_data']['assignee_suggestion'] = f"{a_name} ({a_conf * 100:.0f}%)"
        except Exception as e:
            print(f"⚠️ Parsing Error: {e}")

        # =========================================================
        # [Step 4] Gemini Agent
        # =========================================================
        email_content = data.get('content', '')

        if self.gemini_service.ready:
            print("🤖 [Step 4] Calling Gemini Agents...")
            response['ui_data']['ai_summary'] = self.gemini_service.summarize(email_content)
            response['ui_data']['ai_todo'] = self.gemini_service.extract_todos(email_content)

            if raw_res.get('is_complaint', {}).get('class') == 1:
                response['ui_data']['ai_reply'] = self.gemini_service.generate_reply(
                    email_content,
                    intent="불편을 드려 죄송하며, 담당자가 내용을 확인하고 신속히 연락드리겠다고 안내"
                )
                response['ui_data']['tags'].append("📝답장초안")
        else:
            print("⚠️ Gemini Service is NOT ready. Skipping AI steps.")

        print("✅ Analysis Complete.")
        return response