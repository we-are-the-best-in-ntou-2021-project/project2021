import decodingJason as de
import keras
from keras.models import Sequential,load_model
from keras.layers import Conv1D,MaxPooling1D,BatchNormalization,Flatten,Dense,Activation,Dropout,LSTM,ConvLSTM2D,MaxPooling2D
import numpy as np
from numpy import save
from numpy import load
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

import pylab
import matplotlib.pyplot as plt
import requests

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

name = ["haha","bear","senior","rabbit"]

frame = 50
shift = 3
epochs = 100
offset = True
kernel_size = 3


# c = de.JasonDecoder(dataset_name=action_mix,frame=frame,dirname=dirname,shift=0)
# test,test_labels = c.decoding()
# test_labels = np.array(test_labels)

# test /= 3000

# test = test.reshape(test.shape[0],test.shape[1],test.shape[2]*test.shape[3])
for person in name:
  dirname = "json_data_seperate_" + person
  if not os.path.exists("./model_weight/model.h5"):
    model = load_model("./model_weight/model.h5")

    # for i in range(test.shape[0]):
    #   for j in range(test.shape[1]-1):
    #     for k in range(test.shape[2]):
    #       test[i][j+1][k] -= test[i][0][k]
    #   for j in range(test.shape[2]):
    #     test[i][0][j] = 0

    test_labels = keras.utils.to_categorical(test_labels)
    test_prediction = model.predict(test)
    test_accuracy = sum([np.argmax(test_labels[i]) == np.argmax(test_prediction[i]) for i in range(len(test_labels))])/len(test_labels)
    test_prediction = model.predict_classes(test)
    test_labels = np.argmax(test_labels,axis=1)
    # print(pd.crosstab(test_labels,test_prediction,rownames=['label'],colnames=['predict']))
    # print(test_accuracy)
    print(model.predict(test))
    print(model.predict_classes(test))


  else:

    if os.path.exists(dirname + "/data.npy"):
      data = load(dirname + '/data.npy')
      data_test = load(dirname + '/data_test.npy')
      labels = load(dirname + '/labels.npy')
      labels_test = load(dirname + '/labels_test.npy')
    else:
      a = de.JasonDecoder(dataset_name=action_name,frame=frame,dirname=dirname,shift=shift)
      data, labels = a.decoding()
      labels = np.array(labels)
      save(dirname + '/data.npy',data)
      save(dirname + '/labels.npy',labels)

      b = de.JasonDecoder(dataset_name=action_name_test,frame=frame,dirname=dirname,shift=0)
      data_test, labels_test = b.decoding()
      labels_test = np.array(labels_test)
      save(dirname + '/data_test.npy',data_test)
      save(dirname + '/labels_test.npy',labels_test)

    index=np.arange(data.shape[0])
    np.random.shuffle(index) 
    data = data[index]
    labels = labels[index]

    data = data.reshape(data.shape[0],data.shape[1],data.shape[3]*data.shape[2])
    data_test = data_test.reshape(data_test.shape[0],data_test.shape[1],data_test.shape[3]*data_test.shape[2])
    data /= 3000
    data_test /= 3000
    labels = keras.utils.to_categorical(labels)
    labels_test = keras.utils.to_categorical(labels_test)

    if offset == True:
      for i in range(data.shape[0]):
        for j in range(data.shape[1]-1):
          for k in range(data.shape[2]):
            data[i][j+1][k] -= data[i][0][k]
        for j in range(data.shape[2]):
          data[i][0][j] = 0

      for i in range(data_test.shape[0]):
        for j in range(data_test.shape[1]-1):
          for k in range(data_test.shape[2]):
            data_test[i][j+1][k] -= data_test[i][0][k]
        for j in range(data_test.shape[2]):
          data_test[i][0][j] = 0

    model = Sequential()

    model.add(Conv1D(16,kernel_size=kernel_size,padding='same',activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv1D(32,kernel_size=kernel_size,padding='same',activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())

    # model.add(LSTM(units=10,unroll=True))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    os.makedirs("model_weight",exist_ok=True)
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

    train_history = model.fit(data,labels,batch_size=16,epochs=epochs,verbose=1,validation_split=0.2)
    model.summary()

    model.save("model_weight/model.h5")
    
    score = model.evaluate(data_test,labels_test,verbose=0)
    test_loss = score[0]
    
    predictions = model.predict(data_test)
    test_accuracy = sum([np.argmax(labels_test[i])==np.argmax(predictions[i]) for i in range(len(labels_test))])/len(labels_test)

    predictions = model.predict_classes(data_test)
    labels_test = np.argmax(labels_test,axis=1)

    print(pd.crosstab(labels_test,predictions,rownames=['label'],colnames=['predict']))
    labels_test = keras.utils.to_categorical(labels_test)
    show_train_history(train_history,'accuracy','val_accuracy')

    print("Loss : %s \nAccuracy : %s" % (test_loss,test_accuracy))
    plt.savefig('a.png')

    message = "Name = %s\nEpochs = %s\nKernel_size = %s\nOffset: %s\nLoss : %s \nAccuracy : %s" % (person,epochs,kernel_size,offset,test_loss,test_accuracy)
    r = requests.post(
      f"https://api.telegram.org/bot1628707422:AAEbh-4AJgKXgpiF8f7JOVKq2U0bHR5M5ik/sendMessage",
      json={
        "chat_id":"778687065",
        "text":message,
      },
    )
    