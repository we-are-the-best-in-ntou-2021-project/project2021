# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 18:05:57 2021

@author: hahapiggy

改編自version_2 並嘗試不同的方式加CNN
使用Yoga動作辨識的code: https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1932&context=etd_projects
三個部分
1. 腿
2. 手加軀幹
3. 全部
"""

import JsonDecoder as de
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, TimeDistributed, BatchNormalization
from keras.layers import Conv1D,Flatten, Input, Dropout
from keras.layers.merge import concatenate
from keras.utils import to_categorical
import matplotlib.pyplot as plt

import os


Arm_index = np.array([0,1,2,3,4,5,6,7,8,15,16,17,18])   # 13個關節
Leg_index = np.array([8,9,10,11,12,13,14,19,20,22])     # 10個關節
frame = 50

if os.path.exists("version_4_0323_02"):
    model = load_model("version_4_0323_02")
    print("line 33")
else:
    # 畫出成長曲線的圖
    def show_train_history(train_history, train, validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title("Train History")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="center right")
        # plt.show()
        plt.savefig("version_4_3.png")
        
        
    # 匯入訓練資料
    action_name = ["down", "up", "walk", "run", "raise"]
    a = de.JasonDecoder(dataset_name=action_name, dirname='front_0317' , frame=frame, shift=2)
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
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[3]*X_test.shape[2])
    Y_train = to_categorical(Y_train)
    
    Arm_X_train = X_train[:,:,Arm_index,:]
    
    Leg_X_train = X_train[:,:,Leg_index,:]
    
    
    # 建立模型
    
    def cnn_lstm():
        input_shape = (frame, 25,2)
        arm_input_shape = (frame, 13,2)
        leg_input_shape = (frame, 10,2)
        
        All_X_input = Input(input_shape)
        Arm_input = Input(arm_input_shape)
        Leg_input = Input(leg_input_shape)
        
        All_X = TimeDistributed(Conv1D(filters = 32, kernel_size = 3, activation = 'relu'), input_shape = input_shape)(All_X_input)
        All_X = TimeDistributed(BatchNormalization())(All_X)
        # All_X = TimeDistributed(Conv1D(filters = 32, kernel_size = 3, activation = 'relu'))(All_X)
        # All_X = TimeDistributed(BatchNormalization())(All_X)
        All_X = TimeDistributed(Flatten())(All_X)
        
        Arm_X = TimeDistributed(Conv1D(filters = 32, kernel_size = 3, activation = 'relu'), input_shape = input_shape)(Arm_input)
        Arm_X = TimeDistributed(BatchNormalization())(Arm_X)
        # Arm_X = TimeDistributed(Conv1D(filters = 32, kernel_size = 3, activation = 'relu'))(Arm_X)
        # Arm_X = TimeDistributed(BatchNormalization())(Arm_X)
        Arm_X = TimeDistributed(Flatten())(Arm_X)
        
        Leg_X = TimeDistributed(Conv1D(filters = 32, kernel_size = 3, activation = 'relu'), input_shape = input_shape)(Leg_input)
        Leg_X = TimeDistributed(BatchNormalization())(Leg_X)
        # Leg_X = TimeDistributed(Conv1D(filters = 32, kernel_size = 3, activation = 'relu'))(Leg_X)
        # Leg_X = TimeDistributed(BatchNormalization())(Leg_X)
        Leg_X = TimeDistributed(Flatten())(Leg_X)
        
        merge = concatenate([All_X, Arm_X, Leg_X])
        # merge = All_X
        merge = LSTM(units = 10, unroll=True, return_sequences=True)(merge)
        merge = LSTM(units = 10, unroll=True)(merge)
        merge = Dense(units=5, kernel_initializer='normal', activation='softmax')(merge)
        
        model = Model(inputs = [All_X_input, Arm_input, Leg_input], outputs = merge)
        return model
        
        
        
    # 編譯: 選擇損失函數、優化方法及成效衡量方式
    model = cnn_lstm()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.summary()
    train_history = model.fit(x=[X_train, Arm_X_train, Leg_X_train], y=Y_train, validation_split=0.2, epochs=100, batch_size=50)
    show_train_history(train_history,'acc', 'val_acc')


# 匯入測試資料
action_name_test = ["down_test", "up_test", "walk_test", "run_test", "raise_test"]
b = de.JasonDecoder(dataset_name=action_name_test, dirname='front_test_0317' , frame=frame, shift=2)
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
Y_test = to_categorical(Y_test)

Arm_X_test = X_test[:,:,Arm_index,:]

Leg_X_test = X_test[:,:,Leg_index,:]

results = model.evaluate([X_test, Arm_X_test, Leg_X_test], Y_test, batch_size=128)
print("test loss, test acc:", results)
