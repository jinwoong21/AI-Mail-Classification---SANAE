import pandas as pd
import numpy as np
import pickle
import os
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math

# ==========================================
# 1. 경로 및 설정
# ==========================================
DATA_FILE = '../../data/raw_email_data_date_fixed.csv'
ASSET_DIR = '../../model/jh/'
ONNX_PATH = "../../model/jh/best_model_transformer.onnx"
BATCH_SIZE = 32  # ★ 한 번에 처리할 데이터 개수 (메모리 터지면 줄여라)

print(f"🚀 [Start] ONNX 검증 (Batch Size: {BATCH_SIZE})")
print(f"📂 모델 경로: {os.path.abspath(ONNX_PATH)}")

# ==========================================
# 2. 데이터 로드 및 전처리
# ==========================================
if not os.path.exists(DATA_FILE) or not os.path.exists(ONNX_PATH):
    print("❌ 파일 경로 확인해라.")
    exit()

df = pd.read_csv(DATA_FILE, parse_dates=["received_at"])
if 'is_spam' not in df.columns:
    print("❌ 정답(is_spam) 없다.")
    exit()

# 전처리 (생략 없이 진행)
print("⚙️ 전처리 진행 중...")
df['text'] = (df['title'].astype(str) + "\n" + df['content'].astype(str)).str.strip()
df["received_at"] = pd.to_datetime(df["received_at"])
df["hour"] = df["received_at"].dt.hour
df["dayofweek"] = df["received_at"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
df["text_len_char"] = df["text"].str.len()
df["text_len_word"] = df["text"].str.split().str.len()
df = df.fillna({'text': '', 'sender': '', 'recipient': '', 'has_attachment': 0})

# 자산 로드
with open(f'{ASSET_DIR}/tokenizers.pkl', 'rb') as f:
    loaded_toks = pickle.load(f)
with open(f'{ASSET_DIR}/scalers.pkl', 'rb') as f:
    loaded_scalers = pickle.load(f)

# 시퀀스 변환 (float32 필수)
MAX_LEN_EMAIL = 10
MAX_LEN_TEXT = 200

v_sender = pad_sequences(loaded_toks['sender'].texts_to_sequences(df['sender'].astype(str)), maxlen=MAX_LEN_EMAIL,
                         padding='post').astype(np.float32)
v_recipient = pad_sequences(loaded_toks['recipient'].texts_to_sequences(df['recipient'].astype(str)),
                            maxlen=MAX_LEN_EMAIL, padding='post').astype(np.float32)
v_text = pad_sequences(loaded_toks['text'].texts_to_sequences(df['text'].astype(str)), maxlen=MAX_LEN_TEXT,
                       padding='post').astype(np.float32)

meta_cols = ['has_attachment', 'hour', 'dayofweek', 'is_weekend', 'text_len_char', 'text_len_word']
v_meta = loaded_scalers['transformer'].transform(df[meta_cols].values).astype(np.float32)
y_true = df['is_spam'].fillna(0).astype(int).values

# ==========================================
# 3. ONNX 배치 추론 (핵심 수정)
# ==========================================
print("🧠 ONNX Runtime 추론 시작 (Batch Loop)...")

try:
    sess = ort.InferenceSession(ONNX_PATH)
    output_name = sess.get_outputs()[0].name

    total_samples = len(df)
    y_prob_list = []

    # ★ 배치 단위로 루프 돌리기
    for i in range(0, total_samples, BATCH_SIZE):
        # 슬라이싱 (끝 인덱스는 알아서 처리됨)
        batch_sender = v_sender[i: i + BATCH_SIZE]
        batch_recipient = v_recipient[i: i + BATCH_SIZE]
        batch_text = v_text[i: i + BATCH_SIZE]
        batch_meta = v_meta[i: i + BATCH_SIZE]

        onnx_inputs = {
            'in_sender': batch_sender,
            'in_recipient': batch_recipient,
            'in_text': batch_text,
            'in_meta': batch_meta
        }

        # 부분 추론
        batch_prob = sess.run([output_name], onnx_inputs)[0]
        y_prob_list.append(batch_prob)

        # 진행 상황 (선택 사항)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"   Processed {min(i + BATCH_SIZE, total_samples)}/{total_samples}...")

    # 결과 합치기
    y_prob = np.concatenate(y_prob_list, axis=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    df['prob'] = y_prob
    df['pred'] = y_pred
    print("✅ 추론 완료.")

except Exception as e:
    print(f"\n❌ 추론 중 에러 발생: {e}")
    exit()

# ==========================================
# 4. 성능 평가
# ==========================================
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_mat = confusion_matrix(y_true, y_pred)

print("\n" + "=" * 60)
print(f"📊 [ONNX 최종 검증 결과]")
print("=" * 60)
print(f"✅ Accuracy : {acc:.4f}")
print(f"✅ F1 Score : {f1:.4f}")
print("-" * 60)
print("📌 Confusion Matrix:")
print(f" [[ TN: {conf_mat[0][0]:5d}, FP: {conf_mat[0][1]:5d} ]")
print(f"  [ FN: {conf_mat[1][0]:5d}, TP: {conf_mat[1][1]:5d} ]]")
print("=" * 60)

# 오답 출력 (생략)