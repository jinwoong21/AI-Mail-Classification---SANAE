import json
import time
import csv
from kafka import KafkaProducer
from hdfs import InsecureClient

BOOTSTRAP_SERVERS = ['kafka:9092']
HDFS_URL = 'http://hadoop-master:9870'

TOPIC_NAME = 'email_topic'
HDFS_USER = 'root'
HDFS_FILE_PATH = '/data/final_b2b_emails.csv'
SEND_INTERVAL = 30


def get_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )


def get_hdfs_client():
    return InsecureClient(HDFS_URL, user=HDFS_USER)


def run_producer():
    producer = get_kafka_producer()
    hdfs_client = get_hdfs_client()

    print(f"🚀 [Producer] Connecting to HDFS: {HDFS_FILE_PATH}")

    try:
        with hdfs_client.read(HDFS_FILE_PATH, encoding='utf-8-sig') as reader:
            csv_reader = csv.reader(reader)

            headers = next(csv_reader, None)
            print(f"📋 CSV Headers: {headers}")

            for row in csv_reader:
                if not row:
                    continue

                try:
                    email_data = {
                        'received_at': row[2],
                        'sender': row[3],
                        'subject': row[5],
                        'content': row[6],

                        # raw label
                        'raw_channel': row[8],
                        'raw_dept': row[9],
                        'raw_priority': row[10],
                        'raw_sentiment': row[11],
                        'raw_complaint': row[12],
                        'raw_assignee': row[13],
                        'raw_is_spam': row[14]
                    }

                    # ✅ Kafka 전송 (딱 1번)
                    producer.send(TOPIC_NAME, value=email_data)
                    print(f"✅ [Sent] {email_data['received_at']} | {email_data['subject']}")

                    time.sleep(SEND_INTERVAL)

                except IndexError:
                    continue

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        producer.close()
        print("🛑 Producer finished.")


if __name__ == '__main__':
    run_producer()
