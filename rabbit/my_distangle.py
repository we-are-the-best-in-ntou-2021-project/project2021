from keras import backend as K
from keras.models import Sequential,Model
from keras.layers import Conv1D,MaxPooling1D,BatchNormalization,Dense,Activation,Flatten,Dropout,GlobalAveragePooling1D
from keras.layers import Concatenate,Input,Lambda
from keras.optimizers import Adam
import keras
import numpy as np
from numpy import load,save
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd
import preprocessData as pre
import tensorflow as tf
import datetime

# K.set_floatx('float64')
import GPU_setting as G
G.setting()

def get_data():
  person_name = ["bear","rabbit","haha","senior"]
  dirname = "jsondata"
  epochs = 50
  offset = "none"

  data = load(dirname + '/bear_data.npy')
  labels = load(dirname + '/bear_labels.npy')
  c = np.array(np.full(data.shape[0],0))
  labels = np.vstack((labels,c))
  for i in range(3):
    a = load(dirname + '/'+ person_name[i+1] +'_data.npy')
    b = load(dirname + '/'+ person_name[i+1] +'_labels.npy')
    c = np.array(np.full(a.shape[0],i+1))
    b = np.vstack((b,c))
    data = np.vstack((data,a))
    labels = np.hstack((labels,b))
  
  a = data
  # print(pre.make_velocity(data).shape)
  # data = np.concatenate((data,pre.make_offset(a)),axis=2)
  data = np.concatenate((data,pre.make_velocity(a)),axis=2)
  # data = np.concatenate((data,pre.make_accel(a)),axis=2)
  # print(data.shape)

  data_test = data[np.where(labels[0,:]==0),:,:,:]
  data = data[np.where(labels[0,:]>=3),:,:,:]
  data_test = data_test.reshape(data_test.shape[1],data_test.shape[2],data_test.shape[3]*data_test.shape[4])
  data = data.reshape(data.shape[1],data.shape[2],data.shape[3]*data.shape[4])
  print(data.shape)
  
  # labels = keras.utils.to_categorical(labels)
  # print(labels)
  labels_test = labels[:,np.where(labels[0,:]==0)]
  labels = labels[:,np.where(labels[0,:]>=3)]
  labels_test = labels_test.reshape(labels_test.shape[0],labels_test.shape[2])
  labels = labels.reshape(labels.shape[0],labels.shape[2])

  index=np.arange(data.shape[0])
  np.random.shuffle(index) 
  data = data[index]
  labels[0] = labels[0][index]
  labels[1] = labels[1][index]

  # print(keras.utils.to_categorical(labels[0]))
  # print(keras.utils.to_categorical(labels[1]))
  labels_v = keras.utils.to_categorical(labels[0],num_classes=6)
  labels = keras.utils.to_categorical(labels[1],num_classes=4)
  print(labels_v)
  print(labels)
  labels_test = keras.utils.to_categorical(labels_test[1],num_classes=4)



  return data,data_test,labels,labels_test,labels_v

class GAN():
  def __init__(self,frame=50,dim=100,num_class=4,action_class=6):

    self.latent_dim = 50
    self.num_class = num_class
    self.action_class = action_class
    self.optimizer = Adam(lr=0.00001, beta_1=0.5)
    self.optimizer2 = Adam(lr=0.00001, beta_1=0.5)
    self.input_shape = (frame,dim)
    # self.data_shape = (22016,50,100)
    # self.loss = myloss()

    self.encoder = self.build_encoder()

    self.id_classifer = self.build_ID_classifer()
    self.id_classifer.compile(loss='categorical_crossentropy',optimizer=self.optimizer2,metrics=['accuracy'])
    
    self.action_classifer = self.build_action_classifer()
    self.action_classifer.compile(loss='categorical_crossentropy',optimizer=self.optimizer2,metrics=['accuracy'])

    self.combined = self.build_combine(self.encoder,self.id_classifer,self.action_classifer)
    self.combined.compile(loss=['categorical_crossentropy',self.myloss],optimizer=self.optimizer,loss_weights=[1, 0.01],metrics=['accuracy'])
    self.combined.summary()

  def build_encoder(self,):
  
    model = Sequential()

    model.add(Conv1D(32,kernel_size=7,padding='same',activation='relu'))
    # model.add(Conv1D(64,kernel_size=7,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv1D(64,kernel_size=5,padding='same',activation='relu'))
    # model.add(Conv1D(128,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    
    model.add(Conv1D(128,kernel_size=5,padding='same',activation='relu'))
    # model.add(Conv1D(128,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling1D())
    # model.add(Dropout(rate=0.3))
    model.add(Dense(self.latent_dim))

    return model

  def build_action_classifer(self,):

    model = Sequential()
        
    model.add(Dense(64,input_dim=self.latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(rate=0.3))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))

    # model.add(Dropout(rate=0.5))

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(rate=0.5))

    model.add(Dense(self.action_class, activation='softmax'))

    return model

  def build_ID_classifer(self,):

    model = Sequential()
        
    model.add(Dense(64,input_dim=self.latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(rate=0.3))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))

    # model.add(Dropout(rate=0.5))

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(rate=0.5))

    model.add(Dense(self.num_class, activation='softmax'))

    return model

  def build_combine(self,E,id_C,action_c):

    E.trainable = True
    id_C.trainable = False
    action_c.trainable = False

    input0 = Input(shape=self.input_shape)

    feature = E(input0)

    id_out = id_C(feature)
    action_out = action_c(feature)


    model = Model([input0],[id_out,action_out])
    
    return model

  def myloss(self,y_true,y_pred):
    return tf.negative(K.categorical_crossentropy(y_true,y_pred))

  def random_select(self,data,labels,labels_v,batch_size):
    
    rand_index = np.random.randint(low=0,high=data.shape[0],size=batch_size)
    rand_data0 = data[rand_index,:,:]
    rand_labels = labels[rand_index,:]
    rand_labels_v = labels_v[rand_index,:]

    # print(labels_v.shape)
    # print(labels.shape)

    return rand_data0,rand_labels,rand_labels_v

  def predict_ID(self,x_test,y_test):

    feature = self.encoder.predict(x_test)
    # print(y_test)
    score = self.id_classifer.evaluate(feature,y_test,verbose=1)
    test_loss = score[0]
    predictions = self.id_classifer.predict(feature)
    test_accuracy = sum([np.argmax(y_test[i])==np.argmax(predictions[i]) for i in range(len(y_test))])/len(y_test)

    predictions = self.id_classifer.predict_classes(feature)
    y_test = np.argmax(y_test,axis=1)

    print(pd.crosstab(y_test,predictions,rownames=['label'],colnames=['predict'],normalize='index'))
    print("Loss : %s \nAccuracy : %s" % (test_loss,test_accuracy))

  def train(self,data,labels,data_test,labels_test,labels_v,batch_size,n_batch):

    start_time = datetime.datetime.now()
    zero = np.zeros((batch_size,self.latent_dim))
    for batch in range(n_batch):
      # rand_data,rand_labels,rand_data1,rand_data2 = self.random_select(data,labels,batch_size)

      rand_data,rand_labels,rand_labels_v= self.random_select(data,labels,labels_v,batch_size)
      # rand_labels_v = rand_labels_v.reshape((128,3))
      # rand_labels_v = np.argmax(rand_labels_v)
      loss = self.combined.train_on_batch([rand_data],[rand_labels,rand_labels_v])

      if (batch+1)%1000==0:
        elapsed_time = datetime.datetime.now() - start_time
        print(self.combined.metrics_names)
        print('batch: %d,loss: %s Time: %s'%(batch+1,loss,elapsed_time))
        self.predict_ID(data_test,labels_test)

if __name__ == '__main__':
  x,x_test,y,y_test,y_v = get_data()
  # print(a.shape,c.shape)

  distangle = GAN()
  distangle.train(x,y,x_test,y_test,y_v,batch_size=128,n_batch=40000)
