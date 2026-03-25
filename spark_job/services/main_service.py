import torch
import joblib
import numpy as np
import re
from transformers import AutoTokenizer
# 같은 폴더 내 파일 import
from .model_def import SanaiChainModel


class MainClassifierService:
    def __init__(self, model_path, scaler_path, device="cpu"):
        self.device = torch.device(device)
        self.max_len = 256
        print(f"🤖 Loading PyTorch Model from {model_path}...")

        # 1. 토크나이저 & 스케일러
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

        # 스케일러 로드 (딕셔너리 형태 대응)
        loaded_scaler = joblib.load(scaler_path)
        if isinstance(loaded_scaler, dict):
            self.scaler = loaded_scaler.get('transformer', loaded_scaler.get('ml', list(loaded_scaler.values())[0]))
        else:
            self.scaler = loaded_scaler

        # 2. 모델 초기화
        num_labels = {
            'label_dept': 11, 'mail_channel': 3, 'sentiment': 3,
            'is_complaint': 3, 'priority_level': 4, 'assignee': 3
        }

        self.model = SanaiChainModel("klue/roberta-large", num_labels, feature_dim=7)

        # 3. 가중치 로드
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Main Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Main Model Load Error: {e}")

    def preprocess_one(self, data):
        sender = str(data.get('sender', '')).lower()

        # [수정] 'title'이 없으면 'subject'를 찾도록 변경 (가장 중요!)
        title = str(data.get('title', data.get('subject', '')))

        content = str(data.get('content', ''))
        has_attach = int(data.get('has_attachment', 0))

        # --- Meta Features ---
        domain_score, id_score = 0, 0
        try:
            if '@' in sender:
                user, domain = sender.split('@')
                if 'ssacorp.com' in domain:
                    domain_score = 1
                elif any(x in domain for x in ['gmail', 'naver', 'daum', 'kakao']):
                    domain_score = 0
                else:
                    domain_score = 2

                if '.' in user and not any(c.isdigit() for c in user):
                    id_score = 2
                elif any(r in user for r in ['admin', 'help', 'info']):
                    id_score = 3
                elif any(c.isdigit() for c in user):
                    id_score = 1
        except:
            pass

        is_reply = 1 if re.search(r'^(re|RE|Re|회신):', title) else 0
        is_forward = 1 if re.search(r'^(fw|FW|Fwd|전달):', title) else 0
        text_len_log = np.log1p(len(content))
        thread_depth = 1

        features = np.array([[has_attach, is_reply, is_forward, domain_score, id_score, text_len_log, thread_depth]])

        # 스케일러 적용
        try:
            features_scaled = self.scaler.transform(features)
        except:
            features_scaled = features

        # --- Text Tokenization ---
        full_text = f"[발신] {sender} [제목] {title} [본문] {content}"
        encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return encoding, torch.tensor(features_scaled, dtype=torch.float32)

    def predict(self, data):
        encoding, extras = self.preprocess_one(data)

        with torch.no_grad():
            inputs = {
                'input_ids': encoding['input_ids'].to(self.device),
                'attention_mask': encoding['attention_mask'].to(self.device),
                'extra_features': extras.to(self.device)
            }
            outputs = self.model(**inputs)

        result = {}
        for task, logit in outputs.items():
            probs = torch.softmax(logit, dim=1)
            top_prob, top_class = torch.max(probs, dim=1)
            result[task] = {
                "class": int(top_class.item()),
                "confidence": float(top_prob.item())
            }
        return result