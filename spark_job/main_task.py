from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, struct, to_json
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, BooleanType, TimestampType, \
    LongType
import redis
import json
import psycopg2
import uuid
import datetime
import pandas as pd
import random

# services 폴더 내 모듈 import
try:
    from services.agent_service import EmailAgent
except ImportError:
    EmailAgent = None
    print("⚠️ Agent Service not found, running in lightweight mode.")

# 1. Kafka 설정
KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"
KAFKA_TOPIC = "email_topic"

# 2. DB / Redis 설정
PG_HOST = "app-db"
PG_DB = "mail_logs_db"
PG_USER = "sanae"
PG_PASS = "sanae"

REDIS_HOST = "redis"
REDIS_PORT = 6379


def save_to_sinks(batch_df, batch_id):
    """
    마이크로 배치마다 실행: Kafka 데이터 -> Postgres/Redis 저장
    """
    if batch_df.isEmpty():
        return

    rows = batch_df.collect()

    # DB/Redis 연결
    try:
        pg_conn = psycopg2.connect(host=PG_HOST, database=PG_DB, user=PG_USER, password=PG_PASS)
        pg_cursor = pg_conn.cursor()
        r_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    except Exception as e:
        print(f"❌ DB Connection Failed: {e}")
        return

    print(f"Batch {batch_id}: Processing {len(rows)} emails...")

    for row in rows:
        try:
            # 1. Kafka 데이터 파싱
            data = row.asDict()

            sender_str = data.get('sender', 'Unknown')
            subject = data.get('subject', 'No Subject')
            content = data.get('content', '')
            received_at_str = data.get('received_at')

            # 2. 보낸사람 파싱
            if '<' in sender_str and '>' in sender_str:
                s_name = sender_str.split('<')[0].strip()
                s_email = sender_str.split('<')[1].strip('>')
            else:
                s_name = sender_str
                s_email = sender_str

            # =========================================================
            # 3. [핵심 수정] 한글 변환 제거 -> 코드(숫자) 그대로 저장
            # =========================================================

            # [수정] 한글 변환 제거! CSV에 있는 숫자 코드 그대로 저장해야 함

            # (1) 부서: "개발팀" 변환 로직 삭제 -> raw_dept 그대로 사용
            raw_dept = str(data.get('raw_dept', '0'))

            # (2) 우선순위
            try:
                priority_val = int(data.get('raw_priority', 0))
            except:
                priority_val = 0

            # (3) 감정 (0,1,2 -> 점수)
            try:
                raw_sent = int(data.get('raw_sentiment', 0))

                if raw_sent == 2:  # 긍정
                    sentiment_score = random.randint(75, 98)
                elif raw_sent == 1:  # 부정
                    sentiment_score = random.randint(5, 35)
                else:  # 중립
                    sentiment_score = random.randint(40, 65)

            except:
                sentiment_score = 50

            # (4) 민원 여부
            try:
                is_complaint_bool = (int(data.get('raw_complaint', 0)) >= 1)
            except:
                is_complaint_bool = False

            # (5) 담당자
            try:
                assignee_level = int(data.get('raw_assignee', 0))
            except:
                assignee_level = 0

            # (6) 스팸
            try:
                is_spam_bool = (int(data.get('raw_is_spam', 0)) == 1)
            except:
                is_spam_bool = False

            # main_task.py schema에 추가
            StructField("raw_channel", StringType()),  # ✅ 추가

            # main_task.py 처리부에서
            raw_channel = str(data.get('raw_channel', '0')).strip().split('.')[0]

            # 스팸이면 강제로 0 유지(규칙이 있다면)
            if is_spam_bool:
                ai_category = "0"
            else:
                ai_category = raw_channel  # ✅ 정답 그대로

            # (8) 날짜
            try:
                received_at_dt = pd.to_datetime(received_at_str).to_pydatetime()
            except:
                received_at_dt = datetime.datetime.now()

            # -------------------------------------------------------------
            # PostgreSQL 저장
            # -------------------------------------------------------------
            sql = """
                          INSERT INTO mail_logs (
                              user_email, sender_name, sender_email, 
                              subject, content, received_at, 
                              is_read, is_starred, is_spam, 
                              priority_level, ai_dept, is_complaint, 
                              ai_assignee_level, ai_summary, ai_confidence, 
                              ai_category, sentiment_score, mailbox_type, is_archived
                          ) VALUES (
                              %s, %s, %s, 
                              %s, %s, %s, 
                              %s, %s, %s, 
                              %s, %s, %s, 
                              %s, %s, %s, 
                              %s, %s, %s, %s
                          )
                          """

            pg_cursor.execute(sql, (
                'jaehyun@sanae.com',  # 내 이메일로 고정
                s_name, s_email,
                subject, content, received_at_dt,
                False, False, is_spam_bool,
                priority_val,
                raw_dept,  # [중요] 한글변환 안한 raw값 ("0", "1"...)
                is_complaint_bool,
                assignee_level,
                None, 80,
                ai_category,  # [중요] 숫자 코드 ("0", "1", "2")
                sentiment_score,
                0,  # mailbox_type
                False  # is_archived (이 값을 명시적으로 넣어주는게 안전)
            ))

            # 5. Redis 저장
            redis_data = {
                "sender": s_name,
                "subject": subject,
                "is_spam": is_spam_bool,
                "dept": raw_dept,
                "tags": ai_category
            }
            r_conn.lpush("recent_emails", json.dumps(redis_data))
            r_conn.ltrim("recent_emails", 0, 99)

        except Exception as e:
            print(f"⚠️ Error processing row: {e}")
            continue

    pg_conn.commit()
    pg_cursor.close()
    pg_conn.close()


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("EmailAIProcessing") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    schema = StructType([
        StructField("received_at", StringType()),
        StructField("sender", StringType()),
        StructField("subject", StringType()),
        StructField("content", StringType()),
        StructField("raw_channel", StringType()),
        StructField("raw_dept", StringType()),
        StructField("raw_priority", StringType()),
        StructField("raw_sentiment", StringType()),
        StructField("raw_complaint", StringType()),
        StructField("raw_assignee", StringType()),
        StructField("raw_is_spam", StringType())
    ])

    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    df_parsed = df_raw.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    query = df_parsed.writeStream \
        .foreachBatch(save_to_sinks) \
        .start()

    query.awaitTermination()