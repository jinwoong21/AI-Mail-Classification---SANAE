import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Dropout, \
    LayerNormalization, MultiHeadAttention, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
import tf2onnx
import os


# ==========================================
# 1. 모델 아키텍처 재조립 (Training 코드와 100% 동일해야 함)
# ==========================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    ln1 = LayerNormalization(epsilon=1e-6)
    x = ln1(x + inputs)
    res = x
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    ln2 = LayerNormalization(epsilon=1e-6)
    x = ln2(x + res)
    return x


def build_model():
    # 하이퍼파라미터
    VOCAB_SIZE_EMAIL = 1000
    VOCAB_SIZE_TEXT = 10000
    MAX_LEN_EMAIL = 10
    MAX_LEN_TEXT = 200
    EMBEDDING_DIM = 64

    # 인풋 정의
    in_sender = Input(shape=(MAX_LEN_EMAIL,), name='in_sender')
    emb_sender = Embedding(VOCAB_SIZE_EMAIL, 16)(in_sender)
    flat_sender = GlobalAveragePooling1D()(emb_sender)

    in_recipient = Input(shape=(MAX_LEN_EMAIL,), name='in_recipient')
    emb_recipient = Embedding(VOCAB_SIZE_EMAIL, 16)(in_recipient)
    flat_recipient = GlobalAveragePooling1D()(emb_recipient)

    in_text = Input(shape=(MAX_LEN_TEXT,), name='in_text')
    emb_text = Embedding(VOCAB_SIZE_TEXT, EMBEDDING_DIM)(in_text)

    # 트랜스포머 블록
    transformer_block = transformer_encoder(emb_text, head_size=64, num_heads=4, ff_dim=64, dropout=0.1)
    flat_text = GlobalAveragePooling1D()(transformer_block)

    in_meta = Input(shape=(6,), name='in_meta')
    dense_meta = Dense(32, activation='relu')(in_meta)

    merged = Concatenate()([flat_sender, flat_recipient, flat_text, dense_meta])
    x = Dense(64, activation='relu')(merged)
    bn = BatchNormalization()
    x = bn(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[in_sender, in_recipient, in_text, in_meta], outputs=output)
    return model


# ==========================================
# 2. 모델 생성 및 가중치 로드
# ==========================================
MODEL_PATH = "../../model/jh/best_model_transformer.h5"
ONNX_PATH = "../../model/jh/best_model_transformer.onnx"

print("🔄 모델 껍데기 생성 중...")
model = build_model()

print(f"📥 가중치 로드 중: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"파일 없다: {MODEL_PATH}")

# ★ 핵심: load_model 대신 load_weights 사용 ★
model.load_weights(MODEL_PATH)
print("✅ 모델 복구 완료.")

# ==========================================
# 3. ONNX 변환
# ==========================================
print("🚀 ONNX 변환 시작...")

# 입력 텐서 스펙 정의 (배치 사이즈는 None, 나머지 shape는 모델 정의와 일치)
input_signature = [
    tf.TensorSpec([None, 10], tf.float32, name='in_sender'),
    tf.TensorSpec([None, 10], tf.float32, name='in_recipient'),
    tf.TensorSpec([None, 200], tf.float32, name='in_text'),
    tf.TensorSpec([None, 6], tf.float32, name='in_meta')
]

try:
    # opset 13 정도가 무난함
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=ONNX_PATH
    )
    print(f"🎉 변환 성공! 저장됨: {ONNX_PATH}")

except Exception as e:
    print(f"❌ 변환 실패: {e}")