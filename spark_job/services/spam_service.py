import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime


class SpamService:
    def __init__(self, model_path, tokenizer_path, scaler_path, max_len_email=10, max_len_text=200):
        self.max_len_email = max_len_email
        self.max_len_text = max_len_text

        # Spark Worker CPU 강제
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        print(f"🛡️ Loading Spam Model from {model_path}...")
        try:
            self.model = tf.keras.models.load_model(model_path)

            # 1. 토크나이저 로드 (jh/tokenizers.pkl -> dict: sender, recipient, text)
            print(f"   Loading Tokenizers from {tokenizer_path}...")
            toks = joblib.load(tokenizer_path)
            self.tok_sender = toks.get('sender')
            self.tok_recip = toks.get('recipient')
            self.tok_text = toks.get('text')

            # 2. 스케일러 로드 (jh/scalers.pkl -> dict: lstm, ml, transformer)
            # 학습 때 X_meta를 스케일링했던 'scaler_meta'는 'lstm' 키에 저장되어 있음
            print(f"   Loading Scalers from {scaler_path}...")
            scs = joblib.load(scaler_path)
            self.scaler = scs.get('lstm')  # scaler_meta

        except Exception as e:
            print(f"❌ Spam Model/Tools Load Failed: {e}")
            raise e

    def preprocess_sequence(self, text, tokenizer, max_len):
        """텍스트 -> 시퀀스 -> 패딩"""
        if tokenizer is None:
            # 토크나이저 없으면 0으로 채움 (혹시 모를 오류 방지)
            return np.zeros((1, max_len), dtype=np.float32)

        seq = tokenizer.texts_to_sequences([str(text)])
        padded = np.zeros((1, max_len), dtype=np.float32)

        # 학습 코드 padding='post' 기준
        if len(seq) > 0 and len(seq[0]) > 0:
            length = min(len(seq[0]), max_len)
            padded[0, :length] = seq[0][:length]

        return padded

    def extract_meta_features(self, data):
        """학습 코드와 동일한 6개 메타 피처 추출"""
        # meta_cols = ['has_attachment', 'hour', 'dayofweek', 'is_weekend', 'text_len_char', 'text_len_word']

        content = str(data.get('content', ''))
        has_attach = int(data.get('has_attachment', 0))
        received_at_str = data.get('received_at', '')

        # 시간 파싱
        try:
            dt = pd.to_datetime(received_at_str)
            hour = dt.hour
            dayofweek = dt.dayofweek
            is_weekend = 1 if dayofweek >= 5 else 0
        except:
            hour = 9;
            dayofweek = 0;
            is_weekend = 0

        text_len_char = len(content)
        text_len_word = len(content.split())

        # (1, 6) 형태의 벡터
        features = np.array([[
            has_attach, hour, dayofweek, is_weekend, text_len_char, text_len_word
        ]], dtype=np.float32)

        # 스케일러 적용 (필수)
        if self.scaler:
            try:
                features = self.scaler.transform(features)
            except:
                pass

        return features

    def is_spam(self, data):
        if isinstance(data, str):
            data = {'content': data}

        content = data.get('content', '')
        subject = data.get('subject', data.get('title', ''))
        sender = data.get('sender', '')
        # Kafka 데이터에 recipient가 없으므로 빈 문자열 처리 (모델 입력 shape 유지용)
        recipient = ""

        # -------------------------------------------------------------
        # [중요] 학습 코드의 입력 순서 준수: [Sender, Recipient, Text, Meta]
        # -------------------------------------------------------------

        # 1. Sender (10,)
        X_sender = self.preprocess_sequence(sender, self.tok_sender, self.max_len_email)

        # 2. Recipient (10,)
        X_recipient = self.preprocess_sequence(recipient, self.tok_recip, self.max_len_email)

        # 3. Text (200,)
        X_text = self.preprocess_sequence(content, self.tok_text, self.max_len_text)

        # 4. Meta (6,)
        X_meta = self.extract_meta_features(data)

        try:
            # 4개 입력을 리스트로 전달
            pred = self.model.predict([X_sender, X_recipient, X_text, X_meta], verbose=0)[0][0]
            return pred > 0.5, float(pred)
        except Exception as e:
            print(f"⚠️ Prediction Error: {e}")
            return False, 0.0