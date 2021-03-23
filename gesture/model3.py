# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 04:07:56 2021
C32-L50-L50
@author: User
"""
import decodingJason as de
import keras
from keras.models import Sequential,load_model
from keras.layers import LSTM, Conv1D,MaxPooling1D,BatchNormalization,Flatten,Dense,Activation,Dropout, Reshape
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
import tensorflow as tf
#from sklearn.utils import shuffle
import pylab
import matplotlib.pyplot as plt

import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" #選擇哪一塊gpu
config = ConfigProto()
config.allow_soft_placement=True #如果你指定的設備不存在，允許TF自動分配設備
config.gpu_options.per_process_gpu_memory_fraction=0.9 #分配百分之七十的顯存給程序使用，避免內存溢出，可以自己調整
config.gpu_options.allow_growth = True #按需分配顯存，這個比較重要
session = InteractiveSession(config=config)

action_name = ["down","up","walk","run","raise"]
action_name_test = ["down_test","up_test","walk_test","run_test","raise_test"]
action_mix = ["mix"]

dirname = 'json_data/front'
frame = 50
shift = 1

a = de.JasonDecoder(dataset_name=action_name,frame=frame,dirname=dirname,shift=shift)
data, labels = a.decoding()
labels = np.array(labels)

b = de.JasonDecoder(dataset_name=action_name_test,frame=frame,dirname=dirname,shift=shift)
data_test, labels_test = b.decoding()
labels_test = np.array(labels_test)


c = de.JasonDecoder(dataset_name=action_mix,frame=frame,dirname=dirname,shift=frame)
mix_data,mix_labels = c.decoding()
mix_labels = np.array(mix_labels)


index=np.arange(data.shape[0])
np.random.shuffle(index) 
data = data[index]
labels = labels[index]
# data_test, labels_test = shuffle(data_test, labels_test, random_state = 0)

data = data.reshape(data.shape[0],data.shape[1],data.shape[3]*data.shape[2])
data_test = data_test.reshape(data_test.shape[0],data_test.shape[1],data_test.shape[3]*data_test.shape[2])
mix_data = mix_data.reshape(mix_data.shape[0],mix_data.shape[1],mix_data.shape[2]*mix_data.shape[3])
# data /= 3000
# data_test /= 3000
# data,data_test,labels,labels_test = train_test_split(data,labels,test_size = 0.2,random_state = 10)
labels = keras.utils.to_categorical(labels)
labels_test = keras.utils.to_categorical(labels_test)

#######################################

if os.path.exists("./model_weight/C64_1C128_1L64L32_front.h5"):
  model = load_model("./model_weight/C64_1C128_1L64L32_front.h5")
  train_history = model.fit(data,labels,batch_size=16,epochs=300,verbose=1,validation_split=0.2)
  out = model.predict(mix_data)
  print(model.predict(mix_data))
  print(model.predict_classes(mix_data))
else:
  model = Sequential()
  #C32
  model.add(Conv1D(64,kernel_size=1,padding='valid',activation="relu"))
  model.add(MaxPooling1D(pool_size=2))
  #model.add(BatchNormalization())
  model.add(Conv1D(128,kernel_size=1,padding='same', activation='relu'))
  model.add(BatchNormalization())
  #L32
  model.add(LSTM(batch_input_shape = (None, frame, 128), units = 64, unroll=True, return_sequences=True))
  #L32
  model.add(LSTM(units = 32, unroll=True))

  model.add(Dense(units=5, kernel_initializer='normal', activation='softmax'))

  # model.compile(loss='categorical_crossentropy',
  #               optimizer='adam',
  #               metrics=['accuracy'])
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  os.makedirs("model_weight",exist_ok=True)
  # model.summary()
  print(data.shape,labels.shape)
  print(data_test.shape,labels_test.shape)

  plt.figure(figsize=(24,16),dpi=100)
  def show_train_history(train_history,train,validation):
    plt.ylabel('accuracy')
    plt.legend(['train', 'validation'], loc='center right')
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    
  test_loss = []
  test_accuracy = []

  train_history = model.fit(data,labels,batch_size=16,epochs=300,verbose=1,validation_split=0.2)
  model.save("model_weight/C64_1C128_1L64L32_front.h5")
  
  score = model.evaluate(data_test,labels_test,verbose=0)
  test_loss =score[0]
  
  predictions = model.predict(data_test)
  test_accuracy = sum([np.argmax(labels_test[i])==np.argmax(predictions[i]) for i in range(len(labels_test))])/len(labels_test)

  predictions = model.predict_classes(data_test)
  labels_test = np.argmax(labels_test,axis=1)

  print(pd.crosstab(labels_test,predictions,rownames=['label'],colnames=['predict']))
  labels_test = keras.utils.to_categorical(labels_test)
  show_train_history(train_history,'accuracy','val_accuracy')

  print("Loss : %s \nAccuracy : %s" % (test_loss,test_accuracy))
  plt.savefig('a.png')
  























