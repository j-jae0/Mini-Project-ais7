# Emotions in Text
- 목적 : 텍스트에 내포되어있는 감정을 찾는다.
- 데이터셋 출처 : [Kaggle Dataset - Emotions in text](https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text)

## 1. Emotions_in_text_NLP.ipynb
### (1) 과대적합 이슈 

<div style="text-align : center;">
<img src="https://ifh.cc/g/tJApCX.jpg" width="500">
</div>
<small>Fig 1. 아래 코드에 대한 손실(loss) 및 정확도(accuracy) 그래프</small>

```python
# 1. 모델 구축 (레이어 쌓기)
model = Sequential()
model.add(Embedding(input_dim=vocab_size, 
                     output_dim=embedding_dim, 
                     input_length=max_length))
model.add(Bidirectional(GRU(units=64, return_sequences=True)))
model.add(Bidirectional(GRU(units=64)))
model.add(Dense(units=32, activation='selu'))
model.add(Dense(units=n_class, activation="softmax"))
model.summary()

# 2. 모델 컴파일
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")

# 3. 모델 학습
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train_val, y_train_val
                    , validation_data=(X_valid, y_valid), epochs=100, callbacks=[early_stop])
                    
# 4. 그래프로 확인
df_hist = pd.DataFrame(history.history)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
df_hist[["loss", "val_loss"]].plot(ax=axes[0]).set_title("loss : val_loss")
df_hist[["accuracy", "val_accuracy"]].plot(ax=axes[1]).set_title("accuracy : val_accuracy");
```

### (2) 해결방안 모색
1. 불용어(Stop_words) 제거
2. Tokenizer 파라미터 변경
3. pad_sequences 파라미터 변경
4. train, validation 데이터셋 나눌 때 비율 조절
5. 옵티마이저 변경
6. 활성화함수 변경
7. 모델 레이어 구성 변경(BatchNormalization, Dropout등 층 추가)
+ .. 고민 필요
