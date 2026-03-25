좋아 좋~~다 😎 이제 진짜로 “이메일 AI 플랫폼”의 심장, 모델링을 박아보자.  
말한 것처럼 **똥모델 → 클래식 ML → LSTM 딥러닝** 순서로 진화시키면서,  
나중에 각 기능(부서 분류, 긴급도, 감정, 컴플레인, triage, 스팸 등)에 재사용 가능한 틀로 짤 거야.

일단 오늘은 예시 타겟으로 **label_dept(부서 분류)** 하나 잡고  
전체 파이프라인을 한 번 끝까지 돌려보는 버전으로 갈게.  
그다음 타겟들은 이 틀 그대로 복붙 + 컬럼명만 바꾸면 됨.

---

## 1. 기능별 타겟 정리 (우리가 모델링할 것들)

| 기능                     | 타겟 컬럼        | 문제 타입       | 대표 메트릭            |
|--------------------------|------------------|-----------------|------------------------|
| 자동 분류 및 라우팅      | `label_dept`     | 다중 분류       | accuracy, f1-macro     |
| 긴급도 판단              | `priority_level` | 다중(순서형)    | accuracy, weighted F1  |
| 감정 분석                | `sentiment`      | 다중 분류       | f1-macro               |
| 컴플레인 여부/강도       | `is_complaint`   | 다중(0~3)       | f1-macro, confusion    |
| triage(누가 처리할지)    | `assignee`       | 다중(0~4)       | accuracy, f1-macro     |
| 스팸 여부                | `is_spam`        | 이진 분류       | f1, roc-auc            |

입력(X)은 공통으로:
- 텍스트: `title + content` (공용 텍스트 필드)
- 부가 피처(원하면 나중에 추가): `has_attachment`, `mail_channel`, `received_at` 파생 등

---

## 2. 공용 피처 + 타겟 세팅

아래 코드는 **하나의 파이썬 파일 / 노트북 셀**에 넣고 돌릴 수 있게 짧고 깔끔하게 만들었어.  
(일단 `label_dept` 기준, 나중에 `target_col`만 바꾸면 다른 기능 바로 가능)

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1) 데이터 로드 (이미 df가 있다면 이 부분은 스킵)
df = pd.read_csv('/mnt/data/email_data.csv')  # 경로는 환경에 맞게 수정

# 2) 공용 텍스트 필드 (이미 만들어놨으면 그 컬럼 쓰면 됨)
df['text'] = (df['title'].fillna('') + ' ' + df['content'].fillna('')).str.strip()

# 3) 타겟 설정: 오늘은 label_dept (부서 자동 분류)
target_col = 'label_dept'

# 스팸은 일단 제외하고 정상 메일만 학습
df_model = df[df['is_spam'] == 0].copy()

X_text = df_model['text']
y = df_model[target_col]

# 4) train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

print('train size:', len(X_train), 'test size:', len(X_test))
```

---

## 3. 1단계 – 똥모델 (Baseline: DummyClassifier)

“아무 생각 없음” 수준의 모델로 **현재 데이터 난이도 / 클래스 불균형**을 체크하는 단계야.

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

# feature 없이도 돌아가지만, 형식 맞추려고 일단 리스트로 감싸줌
X_train_dummy = [[0]] * len(X_train)
X_test_dummy = [[0]] * len(X_test)

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_dummy, y_train)
y_pred_dummy = dummy.predict(X_test_dummy)

print('=== Baseline Dummy (most frequent) ===')
print(classification_report(y_test, y_pred_dummy))
```

이거 돌려놓고 나면:
- “아무 생각 없는 모델도 이 정도 정확도다” → 이후 모델이 얼마나 개선되는지 비교 기준.

---

## 4. 2단계 – TF-IDF + 로지스틱 회귀 (클래식 ML 기본기)

이제 진짜 텍스트를 써보자.  
가장 기본이 되는 파이프라인: **TfidfVectorizer + LogisticRegression**.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logi_clf = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,
        ngram_range=(1,2),
        min_df=3
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        C=2.0
    ))
])

logi_clf.fit(X_train, y_train)
y_pred_logi = logi_clf.predict(X_test)

print('=== TF-IDF + LogisticRegression ===')
print(classification_report(y_test, y_pred_logi))
```

여기서부터가 진짜 “쓸만한” 첫 모델이라고 보면 됨.  
Docker에서 CPU만 써도 충분히 빨리 돌아갈 거야.

---

## 5. 3단계 – TF-IDF + LinearSVC (현역 클래스 분류 갓-모델)

텍스트 분류에서 **LinearSVC**는 아직도 현역 에이스급이야.

```python
from sklearn.svm import LinearSVC

svm_clf = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=70000,
        ngram_range=(1,2),
        min_df=3
    )),
    ('clf', LinearSVC())
])

svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

print('=== TF-IDF + LinearSVC ===')
print(classification_report(y_test, y_pred_svm))
```

보통:
- Dummy << LogisticRegression ≲ LinearSVC  
이렇게 계단식으로 올라가는 그림이 나올 거야.

---

## 6. 4단계 – LSTM 딥러닝 모델 (RNN 요건 충족)

이제 프로젝트 필수 스펙인 **RNN(LSTM)**으로 한 번 갈아타보자.  
여기서는 Keras + TensorFlow 기준 예시야.

```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.preprocessing import LabelEncoder

# 1) 라벨 인코딩 (0 ~ num_classes-1)
lbl_enc = LabelEncoder()
y_train_enc = lbl_enc.fit_transform(y_train)
y_test_enc  = lbl_enc.transform(y_test)

num_classes = len(lbl_enc.classes_)
y_train_cat = to_categorical(y_train_enc, num_classes)
y_test_cat  = to_categorical(y_test_enc, num_classes)

# 2) 토크나이저 & 시퀀스 변환
max_words = 30000
max_len   = 200  # 이메일이 길긴 하지만, 우선 200 토큰 정도로 잘라보자

tok = Tokenizer(num_words=max_words, oov_token='<OOV>')
tok.fit_on_texts(X_train)

X_train_seq = tok.texts_to_sequences(X_train)
X_test_seq  = tok.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad  = pad_sequences(X_test_seq,  maxlen=max_len, padding='post', truncating='post')

# 3) LSTM 모델 정의
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# 4) 학습
history = model.fit(
    X_train_pad, y_train_cat,
    validation_split=0.1,
    epochs=5,          # 처음엔 작게 돌려보고 늘리자
    batch_size=256
)

# 5) 평가
test_loss, test_acc = model.evaluate(X_test_pad, y_test_cat, verbose=0)
print(f'=== LSTM Test Accuracy: {test_acc:.4f} ===')
```

이걸로:
- “클래식 ML vs 딥러닝” 비교도 가능하고
- 나중에 **시계열/스레드 기반 LSTM** 같은 변종도 설계할 수 있음 (thread_id 순서대로 정렬해서 학습 등).

---

## 7. 이 구조를 다른 기능에 재사용하는 방법

지금 만든 틀에서 바꿀 건 딱 두 가지야:

1. `target_col`만 바꾸기  
   ```python
   target_col = 'priority_level'   # 긴급도
   # 또는
   target_col = 'sentiment'        # 감정
   target_col = 'is_complaint'     # 컴플레인 강도
   target_col = 'assignee'         # triage
   target_col = 'is_spam'          # 스팸 여부
   ```

2. 이진 분류(`is_spam`)의 경우:
   - `num_classes = 2`로 자동 세팅됨(LabelEncoder로)
   - 메트릭은 accuracy + f1 점수 위주로 보기

나중에 우리 둘이서:

- 한 타겟당: Dummy → TF-IDF+LR → TF-IDF+SVM → LSTM 결과를 한 표에 비교
- 어떤 라벨은 클래식이 더 잘 나오는지, 어떤 라벨은 LSTM이 더 좋은지
- 그리고 “실제 서비스에선 어떤 모델을 쓸지”까지 결정할 수 있음.

---

## 8. 다음 스텝 제안

바로 이어서 이렇게 가면 좋을 듯:

1. 너 환경에서 위 코드로 **label_dept** 한 번 쫙 돌려보기  
2. `classification_report` 결과 찍힌 거 가지고  
   - 어느 라벨이 특히 안 나오는지  
   - 데이터 불균형 영향 있는지  
   같이 해석해보고  
3. 그다음 타겟(`priority_level` or `sentiment`)으로 복붙해서 확장

원하면 다음 턴에:
- **공용 “train_one_target()” 함수** 만들어서  
  `target_col`만 넣으면 위 파이프라인 전부 돌고 리포트 뽑아주는 버전으로 리팩토링도 해줄게.  

일단 여기까지 한 번 꽂아보고, 에러 뜨거나 궁금한 거 나오면 그대로 로그 던져줘 🔧🧠
