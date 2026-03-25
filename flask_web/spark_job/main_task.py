from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, struct, to_json
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
import redis
import json
import psycopg2
import time

# services 폴더 내 모듈 import
from services.agent_service import EmailAgent

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

def save_to_postgres_and_redis(batch_df, batch_id):
    """
    마이크로 배치마다 실행되는 함수
    """
    if batch_df.isEmpty():
        return

    rows = batch_df.collect()

    # [경로 설정] Docker Volume이 마운트된 절대 경로
    base_model_dir = "/opt/spark/model_artifacts"

    try:
        agent = EmailAgent(base_dir=base_model_dir)
    except Exception as e:
        print(f"❌ Agent Init Failed: {e}")
        return

    # DB 연결
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
            # 1. 데이터 준비
            email_data = {
                "sender": row.sender,
                "subject": row.subject,
                "content": row.content,
                "received_at": row.received_at,
                "has_attachment": 0
            }

            # 2. Agent 추론
            result = agent.process_email(email_data)

            # 3. 결과 파싱
            is_spam = (result['type'] == 'SPAM')
            spam_prob = result.get('confidence', 0.0)

            ui_data = result.get('ui_data', {})
            dept = ui_data.get('folder', '기타')
            tags = ",".join(ui_data.get('tags', []))
            assignee = ui_data.get('assignee_suggestion', '')

            # 4. PostgreSQL 저장
            sql = """
                  INSERT INTO mail_logs
                  (received_at, sender, subject, content, is_spam, spam_prob, department, assignee_name)
                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                  """
            pg_cursor.execute(sql, (
                email_data['received_at'], email_data['sender'], email_data['subject'], email_data['content'],
                is_spam, spam_prob, dept, assignee
            ))

            # 5. Redis 저장
            redis_data = {
                "sender": email_data['sender'],
                "subject": email_data['subject'],
                "is_spam": is_spam,
                "dept": dept,
                "tags": tags
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
    spark = SparkSession.builder.appName("EmailAIProcessing").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    schema = StructType([
        StructField("received_at", StringType()),
        StructField("sender", StringType()),
        StructField("subject", StringType()),
        StructField("content", StringType())
    ])

    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    df_parsed = df_raw.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    query = df_parsed.writeStream \
        .foreachBatch(save_to_postgres_and_redis) \
        .start()

    query.awaitTermination()