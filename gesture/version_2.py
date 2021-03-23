# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:25:58 2021

@author: hahapiggy
放入新的dataset
"""

import JsonDecoder as de
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
# 畫出成長曲線的圖
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="center right")
    # plt.show()
    plt.savefig("version_2_train_history_9.png")
    

# 匯入訓練資料
action_name = ["down", "up", "walk", "run", "raise"]
frame = 50
a = de.JasonDecoder(dataset_name=action_name, dirname='front_0305' , frame=frame, shift=2)
X_train, Y_train = a.decoding()
Y_train = np.array(Y_train)

# 洗亂順序，不然按照原本的順序無法訓練(全是0後全是1...)
X_train, Y_train = shuffle(X_train, Y_train, random_state = 0)

for i in range(X_train.shape[0]):
    for j in range(X_train.shape[1]):
        scaler = MinMaxScaler()
        X_train[i,j] = scaler.fit_transform(X_train[i][j])

# 原本data為: (幾筆data, frame, node=25, XY=2)     
# 現在將data轉換為: (幾筆data, frame, 50) -> 因為要將x和y拆開來看 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[3]*X_train.shape[2])
Y_train = to_categorical(Y_train)


# 建立模型
model = Sequential()

model.add(LSTM(batch_input_shape = (None, frame, 50), units = 64, unroll=True, return_sequences=True))
model.add(LSTM(units = 64, unroll=True))

model.add(Dense(units=5, kernel_initializer='normal', activation='softmax'))
# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.summary()
train_history = model.fit(x=X_train, y=Y_train, validation_split=0.2, epochs=100, batch_size=50,)
show_train_history(train_history,'acc', 'val_acc')


# 匯入測試資料
action_name_test = ["down_test", "up_test", "walk_test", "run_test", "raise_test"]
b = de.JasonDecoder(dataset_name=action_name_test, dirname='front_test_0305' , frame=frame, shift=2)
X_test, Y_test = b.decoding()
Y_test = np.array(Y_test)

# 洗亂順序，不然按照原本的順序無法訓練(全是0後全是1...)
X_test, Y_test = shuffle(X_test, Y_test, random_state = 0)

for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        scaler = MinMaxScaler()
        X_test[i,j] = scaler.fit_transform(X_test[i][j])
        
# 原本data為: (幾筆data, frame, node=25, XY=2)
# 現在將data轉換為: (幾筆data, frame, 50) -> 因為要將x和y拆開來看 
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[3]*X_test.shape[2])
Y_test = to_categorical(Y_test)

results = model.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)