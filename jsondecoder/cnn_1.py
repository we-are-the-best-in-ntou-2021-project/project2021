import decodingJason as de
import keras
from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,BatchNormalization,Flatten,Dense,Activation,Dropout
import numpy as np
from sklearn.model_selection import train_test_split

import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" #選擇哪一塊gpu
config = ConfigProto()
config.allow_soft_placement=True #如果你指定的設備不存在，允許TF自動分配設備
config.gpu_options.per_process_gpu_memory_fraction=0.8 #分配百分之七十的顯存給程序使用，避免內存溢出，可以自己調整
config.gpu_options.allow_growth = True #按需分配顯存，這個比較重要
session = InteractiveSession(config=config)


action_name = ["down","up","walk","run","raise","phone"]
# action_name_test = ["down_test","up_test","walk_test","run_test","raise_test","phone_test"]

a = de.JasonDecoder(dataset_name=action_name,frame=100,dirname="dataset",shift=10)
data, labels = a.decoding()
labels = np.array(labels)

# b = de.JasonDecoder(dataset_name=action_name,frame=100,shift=10)
# data_test, labels_test = b.decoding()
# labels_test = np.array(labels_test)

data = data.reshape(data.shape[0],data.shape[1],data.shape[3]*data.shape[2])
# data_test = data_test.reshape(data_test.shape[0],data_test.shape[1],data_test.shape[3]*data_test.shape[2])
# data /= 3000
data,data_test,labels,labels_test = train_test_split(data,labels,test_size = 0.2,random_state = 10)
labels = keras.utils.to_categorical(labels)
labels_test = keras.utils.to_categorical(labels_test)


model = Sequential()
model.add(Conv1D(16,kernel_size=3,padding='same',activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv1D(32,kernel_size=3,padding='same',activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(data.shape,labels.shape)
print(data_test.shape,labels_test.shape)

train_history = model.fit(data,labels,batch_size=16,epochs=50,verbose=1,validation_split=0.2)
score = model.evaluate(data_test,labels_test,verbose=1)

print('Test loss: ',score[0])
print('Test accuracy: ',score[1])
# model.summary()
          

