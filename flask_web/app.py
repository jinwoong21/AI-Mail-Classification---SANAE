import os
import sys
import redis
import random
import subprocess
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# ---------------------------------------------------------
# [설정] Spark & Producer 실행 명령어 (도커 내부 경로 기준)
# ---------------------------------------------------------
PRODUCER_CMD = ["python3", "/opt/spark/semi-prj-2/ingestion/producer.py"]
SPARK_CMD = [
    "/opt/spark/bin/spark-submit",
    "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
    "/opt/spark/semi-prj-2/spark_job/main_task.py"
]

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'sanae-secret-key-1234'

# ---------------------------------------------------------
# [설정] DB 및 Redis
# ---------------------------------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://sanae:sanae@app-db:5432/mail_logs_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Redis 연결
try:
    redis_client = redis.StrictRedis(host='redis', port=6379, db=0, decode_responses=True)
    redis_status = "✅ Connected"
except Exception:
    redis_client = None
    redis_status = "❌ Not Connected"

DEPT_MAP = {
    "0": "개발팀", "1": "서비스팀", "2": "영업팀", "3": "마케팅팀",
    "4": "경영팀", "5": "인사팀", "6": "기획팀", "7": "운영팀",
    "8": "재무팀", "9": "보안팀", "10": "디자인팀"
}

CHANNEL_MAP = {
    "0": "일반",
    "1": "사내업무",
    "2": "거래처"
}

# ---------------------------------------------------------
# [모델] Email 테이블 정의 (최종 수정본 적용)
# ---------------------------------------------------------
class Email(db.Model):
    __tablename__ = 'mail_logs'

    id = db.Column(db.Integer, primary_key=True)

    # 사용자 및 보낸사람 정보
    user_email = db.Column(db.String(100), nullable=False)
    sender_name = db.Column(db.String(100))
    sender_email = db.Column(db.String(100))

    subject = db.Column(db.String(255))
    content = db.Column(db.Text)
    received_at = db.Column(db.DateTime, default=datetime.now)

    # 상태 플래그
    is_read = db.Column(db.Boolean, default=False)
    is_starred = db.Column(db.Boolean, default=False)

    # [🔥핵심 수정] 이 줄이 없어서 에러가 났던 겁니다!
    is_archived = db.Column(db.Boolean, default=False)

    mailbox_type = db.Column(db.Integer, default=0, server_default=text('0'))

    # AI 분석 결과
    is_spam = db.Column(db.Boolean, default=False)
    is_complaint = db.Column(db.Boolean, default=False)

    priority_level = db.Column(db.Integer, default=0)
    ai_dept = db.Column(db.String(50))
    ai_assignee_level = db.Column(db.Integer, default=0)

    ai_summary = db.Column(db.Text, nullable=True)
    ai_confidence = db.Column(db.Integer, default=0)

    # 추가 기능
    ai_category = db.Column(db.String(50))
    sentiment_score = db.Column(db.Integer, default=50)

    # ------------------------------------------------------------------
    # [화면 표시용 변환 로직] (한글 매핑을 위해 필수)
    # ------------------------------------------------------------------
    @property
    def display_dept(self):
        # 상단 DEPT_MAP 변수 사용
        return DEPT_MAP.get(str(self.ai_dept), "기타부서")

    @property
    def display_channel(self):
        # 상단 CHANNEL_MAP 변수 사용
        return CHANNEL_MAP.get(str(self.ai_category), "일반")

    @property
    def display_priority(self):
        if self.priority_level == 0: return "낮음"
        if self.priority_level == 1: return "일반"
        return "🚨긴급"

    @property
    def display_assignee(self):
        if self.ai_assignee_level <= 1: return "실무자"
        if self.ai_assignee_level <= 3: return "관리자"
        return "책임자"

    @property
    def display_sentiment(self):
        val = self.sentiment_score
        if val <= 30: return "😡부정"
        if val >= 70: return "😊긍정"
        return "😐중립"

    @property
    def display_complaint(self):
        if self.is_complaint: return "🔥강함"
        return "없음"


# ---------------------------------------------------------
# [AI 모듈 로드]
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
agent = None
gemini_agent = None

try:
    from spark_job.services.agent_service import EmailAgent

    agent = EmailAgent(BASE_DIR)
    print("✅ EmailAgent Imported")
except Exception as e:
    print(f"❌ Agent Import Failed: {e}")

try:
    from spark_job.services.gemini_agents import get_gemini_service

    gemini_agent = get_gemini_service(BASE_DIR)
    print("✅ Gemini Service Imported")
except Exception as e:
    print(f"❌ Gemini Service Failed: {e}")


# ---------------------------------------------------------
# [유틸] 템플릿 필터
# ---------------------------------------------------------
@app.template_filter('relative_time')
def relative_time(dt):
    if not dt: return ""
    now = datetime.now()
    diff = now - dt
    if diff.days == 0:
        return dt.strftime("%H:%M")
    elif diff.days == 1:
        return "어제"
    elif diff.days < 7:
        return f"{diff.days}일 전"
    else:
        return dt.strftime("%Y.%m.%d")


# ---------------------------------------------------------
# [API] AI 분석 (JSON 통신)
# ---------------------------------------------------------
@app.route('/api/analyze', methods=['POST'])
def analyze_email():
    if agent is None:
        return jsonify({"status": "ERROR", "message": "AI Model not loaded"}), 500
    try:
        data = request.json
        if not data or 'content' not in data:
            return jsonify({"status": "ERROR", "message": "No content provided"}), 400
        result = agent.process_email(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500


# ---------------------------------------------------------
# [라우트] 메인 및 메일함 조회
# ---------------------------------------------------------
@app.route('/')
@app.route('/main')
def main():
    user_email = get_current_user_email() or 'jaehyun@sanae.com'

    # -----------------------------
    # (1) 받은메일함 기본 쿼리
    # -----------------------------
    base_q = Email.query.filter(
        Email.user_email == user_email,
        Email.is_spam.is_(False),
        Email.is_archived.is_(False),
        Email.mailbox_type == 0
    )

    # -----------------------------
    # (2) 중요 메일 브리핑 (기존)
    # -----------------------------
    important_mails = (base_q
        .filter((Email.priority_level >= 1) | (Email.is_complaint.is_(True)))
        .order_by(Email.received_at.desc())
        .limit(8)
        .all()
    )

    urgent_count    = base_q.filter(Email.priority_level >= 1).count()
    critical_count  = base_q.filter(Email.priority_level >= 2).count()
    complaint_count = base_q.filter(Email.is_complaint.is_(True)).count()

    # -----------------------------
    # (3) ✅ 도넛용: 받은메일함 전체 기준 "미처리(안읽음)"
    # -----------------------------
    total_inbox = base_q.count()
    unread_inbox = base_q.filter(Email.is_read.is_(False)).count()

    urgent_unread = base_q.filter(
        Email.is_read.is_(False),
        (Email.priority_level >= 1) | (Email.is_complaint.is_(True))
    ).count()

    # ✅ 처리율(읽음 완료율): 처리할수록 100%로
    completion_rate = 100 if total_inbox == 0 else int(round(((total_inbox - unread_inbox) / total_inbox) * 100))

    # -----------------------------
    # (4) 템플릿으로 넘기기
    # -----------------------------
    return render_template(
        'main.html',
        important_mails=important_mails,
        urgent_count=urgent_count,
        critical_count=critical_count,
        complaint_count=complaint_count,

        total_inbox=total_inbox,
        unread_inbox=unread_inbox,
        urgent_unread=urgent_unread,
        completion_rate=completion_rate
    )


@app.route('/mail_inbox')
def mail_inbox():
    # ▼▼▼ [긴급 처방] 이 줄을 추가하세요! (무조건 덮어쓰기) ▼▼▼
    session['user'] = 'jaehyun@sanae.com'

    # ... (그 다음 코드는 원래대로)
    user_email = session['user']
    # ...

    # 🔥🔥🔥 디버그 출력 🔥🔥🔥
    print("===================================")
    print("[DEBUG] session user_email:", user_email)
    print("[DEBUG] TOTAL MAILS:", Email.query.count())
    print("[DEBUG] USER MAILS:",
          Email.query.filter(Email.user_email == user_email).count())
    print("[DEBUG] SPAM FALSE:",
          Email.query.filter(Email.is_spam.is_(False)).count())
    print("[DEBUG] ARCHIVED FALSE:",
          Email.query.filter(Email.is_archived.is_(False)).count())
    print("===================================")

    def _severity_rank(e):
        p = int(e.priority_level or 0)
        c = bool(e.is_complaint)

        # 🔴 빨강: (매우긴급) or (긴급 + 민원)
        if p >= 2 or (p == 1 and c):
            return 2
        # 🟠 주황: (긴급) or (민원)
        if p == 1 or c:
            return 1
        # ⚪ 흰색: 그 외
        return 0

    emails = Email.query.filter(
        Email.user_email == user_email,
        Email.is_spam.is_(False),
        Email.is_archived.is_(False),
        Email.mailbox_type == 0
    ).order_by(Email.received_at.desc()).all()

    # ✅ 핵심: 빨강→주황→흰색 우선, 같은 그룹 내에서는 최신순
    emails = sorted(
        emails,
        key=lambda e: (_severity_rank(e), e.received_at or datetime.min),
        reverse=True
    )

    return render_template(
        'mail_inbox.html',
        emails=emails,
        urgent_count=len(emails)
    )



@app.route('/mail_spam')
def mail_spam():
    try:
        # 스팸함 (is_spam=True)
        spam_mails = Email.query.filter_by(is_spam=True).order_by(Email.received_at.desc()).all()
    except:
        spam_mails = []
    return render_template('mail_spam.html', spam_mails=spam_mails)


@app.route('/mail_trash')
def mail_trash():
    try:
        # [수정] 휴지통 (mailbox_type=2)
        trash_mails = Email.query.filter_by(mailbox_type=2).order_by(Email.received_at.desc()).all()
    except:
        trash_mails = []
    # 휴지통 템플릿 재사용 (필요시 mail_trash.html 생성)
    return render_template('mail_inbox.html', emails=trash_mails)


@app.route('/mail_drafts')
def mail_drafts():
    try:
        # [수정] 보관함 (mailbox_type=1)
        drafts = Email.query.filter_by(mailbox_type=1).order_by(Email.received_at.desc()).all()
    except:
        drafts = []
    return render_template('mail_drafts.html', emails=drafts)


# ---------------------------------------------------------
# [라우트] 메일 상세보기 & 쓰기 (기존 기능 유지)
# ---------------------------------------------------------
@app.route('/mail_detail/<int:email_id>')
def mail_detail(email_id):
    email = Email.query.get_or_404(email_id)
    if not email.is_read:
        email.is_read = True
        db.session.commit()
    return render_template('mail_detail.html', email=email)


@app.route('/mail_write')
def mail_write():
    # 답장하기 로직 (reply_to)
    to_addr, subject, content = "", "", ""
    reply_id = request.args.get('reply_to')

    if reply_id:
        try:
            original = Email.query.get(reply_id)
            if original:
                to_addr = original.sender_email
                subject = f"Re: {original.subject}" if not original.subject.lower().startswith(
                    "re:") else original.subject
                sent_time = original.received_at.strftime('%Y-%m-%d %H:%M')
                content = f"\n\n\n-----Original Message-----\nFrom: {original.sender_name}\nSent: {sent_time}\nSubject: {original.subject}\n\n{original.content}"
        except:
            pass

    return render_template('mail_write.html', to_addr=to_addr, subject=subject, content=content)


# [기존] 보낸 메일함
sent_mailbox = []


@app.route('/mail_sent')
def mail_sent():
    return render_template('mail_sent.html', sent_mails=sent_mailbox)


@app.route('/mail_sent_view/<int:idx>')
def mail_sent_view(idx):
    if idx < 0 or idx >= len(sent_mailbox):
        return "메일이 없습니다.", 404

    selected_mail = sent_mailbox[idx]
    return render_template("mail_sent_view.html", email=selected_mail)  # ✅ email로 넘겨야 함




# ---------------------------------------------------------
# [API] 메일 전송 (기존 기능 유지)
# ---------------------------------------------------------
@app.route('/send_reply', methods=['POST'])
def send_reply():
    sent_mailbox.append({
        "sender": request.form.get('from_email'),
        "receivers": request.form.get('to_email'),
        "subject": request.form.get('subject'),
        "content": request.form.get('body')
    })
    return render_template("mail_inbox_done.html", subject=request.form.get('subject'))


sent_mailbox = []

@app.route('/send_from_write', methods=['POST'])
def send_from_write():
    user_email = session.get('user', 'jaehyun@sanae.com')
    sent_mailbox.append({
        "received_at": datetime.now(),  # ✅ datetime
        "sender_email": user_email,
        "sender_name": user_email.split("@")[0],
        "receivers": request.form.get('receivers'),
        "cc": request.form.get('cc'),
        "subject": request.form.get('subject'),
        "content": request.form.get('content'),
    })
    return redirect(url_for('mail_sent'))


# ---------------------------------------------------------
# [API] 메일 이동/삭제 (DB 연동)
# ---------------------------------------------------------
@app.route('/api/move_mails', methods=['POST'])
def move_mails():
    """ 선택한 메일들을 보관함(1) 또는 휴지통(2)으로 이동 """
    data = request.json
    mail_ids = data.get('ids', [])
    target_type = data.get('target_type')

    if not mail_ids or target_type is None:
        return jsonify({"status": "error", "message": "잘못된 요청"})

    try:
        db.session.query(Email).filter(Email.id.in_(mail_ids)).update(
            {Email.mailbox_type: int(target_type)}, synchronize_session=False
        )
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)})


# ---------------------------------------------------------
# [API] 파이프라인 제어 (Start / Stop / Status)
# ---------------------------------------------------------
@app.route('/api/start_pipeline', methods=['POST'])
def start_pipeline():
    try:
        subprocess.run(["pkill", "-f", "producer.py"])
        subprocess.run(["pkill", "-f", "main_task.py"])
        subprocess.Popen(SPARK_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen(PRODUCER_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("🚀 [System] Pipeline Started")
        return jsonify({"status": "success", "message": "파이프라인 가동됨"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/stop_pipeline', methods=['POST'])
def stop_pipeline():
    try:
        subprocess.run(["pkill", "-f", "producer.py"])
        subprocess.run(["pkill", "-f", "main_task.py"])
        print("🛑 [System] Pipeline Stopped")
        return jsonify({"status": "success", "message": "파이프라인 중지됨"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/pipeline_status', methods=['GET'])
def check_pipeline_status():
    try:
        result = subprocess.run(["pgrep", "-f", "producer.py"], stdout=subprocess.PIPE)
        return jsonify({"status": "running" if result.returncode == 0 else "stopped"})
    except:
        return jsonify({"status": "error"})


# ---------------------------------------------------------
# [API] Gemini 기능 (요약, 할일, 답장 등)
# ---------------------------------------------------------
@app.route('/api/gemini/execute', methods=['POST'])
def execute_gemini():
    if not gemini_agent or not gemini_agent.ready:
        return jsonify({"status": "ERROR", "message": "Gemini Not Ready"}), 503

    data = request.json
    action = data.get('action')
    text = data.get('text', '')

    try:
        result = ""
        if action == 'summary':
            result = gemini_agent.summarize(text)
        elif action == 'todo':
            result = gemini_agent.extract_todos(text)
        elif action == 'reply':
            result = gemini_agent.generate_reply(text, data.get('intent', ''))
        elif action == 'fix':
            result = gemini_agent.fix_email(text)
        elif action == 'translate':
            result = gemini_agent.translate(text)

        print(f"[GEMINI] action={action} text_len={len(text)}")
        head = (text[:120] if text else "").replace("\n", " ")
        print(f"[GEMINI] text_head={head}")

        # result 나온 뒤에
        print(f"[GEMINI] result_len={len(result)} result_head={result[:120]}")

        return jsonify({"status": "SUCCESS", "result": result})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500


# ---------------------------------------------------------
# User 모델 (signup_user 테이블 매핑)
# ---------------------------------------------------------
class User(db.Model):
    __tablename__ = 'signup_user'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100))
    role = db.Column(db.String(50), default='USER')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime)

# -------------------------------
# 로그인 / 회원가입 페이지 (postgre)
# -------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        # DB에서 사용자 조회
        user = User.query.filter_by(email=email, password=password).first()
        # 로그인 실패
        if not user:
            return render_template(
                'login.html',
                error="이메일 또는 비밀번호가 올바르지 않습니다."
            )
        # 로그인 성공 → 세션 저장
        session['user'] = {
            'id': user.id,
            'email': user.email,
            'name': user.name,
            'role': user.role
        }
        print("✅ LOGIN SESSION SET:", session['user'])  # ← 추가
        return redirect(url_for('main'))
    return render_template('login.html')

def get_current_user_email():
    u = session.get('user')
    if isinstance(u, dict):
        return u.get('email')
    if isinstance(u, str):
        return u
    return None

@app.route('/api/urgent_status', methods=['GET'])
def urgent_status():
    user_email = get_current_user_email()
    if not user_email:
        return jsonify({"status": "error", "message": "not_logged_in"}), 401

    base_filter = (
        (Email.user_email == user_email) &
        (Email.is_spam.is_(False)) &
        (Email.is_archived.is_(False)) &
        (Email.mailbox_type == 0) &         # 받은메일함
        (Email.is_read.is_(False)) &
        (Email.priority_level >= 1)         # ✅ 긴급(1) 이상 (critical=2~)
    )

    count = Email.query.filter(base_filter).count()

    # ✅ desc() import 없이 정렬: 컬럼의 .desc() 메서드 사용
    latest = (Email.query
              .filter(base_filter)
              .order_by(Email.received_at.desc())
              .limit(5)
              .all())

    items = [{
        "id": e.id,
        "subject": e.subject or "",
        "sender": getattr(e, "sender_name", None) or getattr(e, "sender", "") or "",
        "priority": int(e.priority_level or 0),
        "received_at": e.received_at.isoformat() if e.received_at else ""
    } for e in latest]

    return jsonify({"status": "success", "count": count, "items": items})



@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/test2/gemini')
def test_gemini_page():
    return render_template('test2.html')


@app.route('/debug_db')
def debug_db():
    try:
        result = db.session.execute(text("SELECT * FROM mail_logs ORDER BY id DESC"))
        rows = [dict(row._mapping) for row in result]
        return jsonify({"status": "SUCCESS", "count": len(rows), "rows": rows})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)})


@app.route('/reset_db')
def reset_db():
    try:
        db.session.execute(text('DROP TABLE IF EXISTS mail_logs CASCADE'))
        db.session.commit()
        db.create_all()
        return "✅ DB 초기화 완료! (테이블 재생성됨)"
    except Exception as e:
        return f"❌ 초기화 실패: {e}"

@app.route('/debug_columns')
def debug_columns():
    result = db.session.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'mail_logs'
        ORDER BY column_name
    """))
    return jsonify([row[0] for row in result])


@app.route('/api/search_mails')
def search_mails():
    query = request.args.get('q', '')

    # DB에서 제목(subject)에 검색어가 포함된 메일 최대 10개 추출
    # 예: SELECT * FROM emails WHERE subject LIKE '%query%' LIMIT 10
    results = Email.query.filter(Email.subject.contains(query)).limit(10).all()

    mail_list = []
    for m in results:
        mail_list.append({
            "id": m.id,
            "subject": m.subject,
            "sender_name": m.sender_name,
            "received_at": m.received_at.strftime('%Y-%m-%d')
        })

    return jsonify({
        "status": "success",
        "mails": mail_list
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)