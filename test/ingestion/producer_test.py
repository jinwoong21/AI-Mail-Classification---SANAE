import time
import json
import pandas as pd
import os
from kafka import KafkaProducer

# ==========================================
# 1. 설정 (경로 수정됨)
# ==========================================
BOOTSTRAP_SERVERS = ['localhost:19092']  # 로컬 테스트용 포트
TOPIC_NAME = 'email_topic'

# 현재 파일(producer_test.py)의 위치를 기준으로 데이터 파일 절대 경로 계산
# 구조: semi-prj-2/test/ingestion/producer_test.py -> ../../data/파일
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 폴더 (test/ingestion)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) # 루트 폴더 (semi-prj-2)
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw_email_data_date_fixed.csv')

# ==========================================
# 2. Kafka Producer 연결
# ==========================================
print(f"🔌 Kafka 연결 시도 중... ({BOOTSTRAP_SERVERS})")
try:
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    print("✅ Kafka 연결 성공!")
except Exception as e:
    print(f"❌ Kafka 연결 실패: {e}")
    print("팁: docker ps로 kafka 컨테이너(포트 19092) 켜져 있는지 확인해.")
    exit()

# ==========================================
# 3. 데이터 발사 (테스트 로직)
# ==========================================
def run_test_producer():
    print(f"📂 CSV 파일 찾는 중... \n -> {CSV_PATH}")
    
    if not os.path.exists(CSV_PATH):
        print("❌ CSV 파일이 없다! 경로 확인해라.")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        # 날짜 타입 이슈 방지 (문자열 변환)
        if 'received_at' in df.columns:
            df['received_at'] = df['received_at'].astype(str)
            
        print(f"🚀 [TEST] 데이터 발사 시작 (총 {len(df)}건 중 10건만 테스트)")
        print("-" * 50)

        # 테스트니까 딱 10개만 보내고 종료 (무한 루프 X)
        for idx, row in df.iterrows():
            if idx >= 10: 
                print("🛑 테스트 종료 (10건 전송 완료)")
                break
                
            data = row.to_dict()
            producer.send(TOPIC_NAME, value=data)
            
            print(f"[Sent] No.{idx} -> {data.get('title', 'No Title')[:20]}...")
            time.sleep(1) # 1초 대기

        producer.flush()
        print("🎉 테스트 성공! Kafka Consumer에서 데이터 들어왔나 확인해봐.")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == '__main__':
    run_test_producer()