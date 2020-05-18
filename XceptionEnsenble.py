#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import cv2, sys, re
import pandas as pd
import numpy as np
import random
from keras import models,Sequential,layers
from keras.models import Model, Input
from keras.layers import Conv2D, SeparableConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Dropout, Flatten, Reshape
from keras.layers import Add
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.optimizers import SGD,Adam
from keras.regularizers import l2
from keras.datasets import cifar10
import numpy as np
import keras.backend as K
from sklearn.metrics import log_loss
import glob
from keras.models import load_model
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf

from keras import backend as K
from keras.models import Model

from keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Softmax, Flatten

# ** to update custom Activate functions
from keras.utils.generic_utils import get_custom_objects


# In[ ]:


# Numpy 데이터 불러오기 (이미지, 라벨이 묶여서 저장되어 있음)
arr = np.load('/kaggle/input/newdata/new_data_side.npz','r')

X = arr['arr_0']
y = arr['arr_1']

# X = np.r_[arr['arr_0'][:10000],arr['arr_0'][13000:23000]]
# y = np.r_[arr['arr_1'][:10000],arr['arr_1'][13000:23000]]

# 메모리 줄이기 위해 사용
del arr

# Data Shuffle

s = np.arange(X.shape[0])
np.random.shuffle(s)

X = X[s]
y = y[s]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=321)

del X
del y


# In[9]:


# cBam(SelfAttention)을 추가할 때 사용

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid


def cbam_block(cbam_feature, ratio=8):
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature, ratio=8):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 1024
	
	if K.image_data_format() == "channels_first":
		channel = input_feature._keras_shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature._keras_shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature._keras_shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])
# ----------------------------------------------------------


# In[11]:


# Xception Model

def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu', weight_decay=1e-5):
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x

def sepconv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu', weight_decay=1e-5,
                 depth_multiplier=1):
    x = SeparableConv2D(filters, kernel_size, padding=padding, strides=strides, depth_multiplier=depth_multiplier,
                        depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)

    if activation:
        x = Activation(activation)(x)
    return x


def Xception(model_input, classes):
    ## Entry flow
    x = conv2d_bn(model_input, 32, (3, 3), strides=2) 
    x = conv2d_bn(x, 64, (3, 3))

    for fliters in [156, 312, 1024]: 
        residual = conv2d_bn(x, fliters, (1, 1), strides=2, activation=None)

        x = Activation(activation='relu')(x)
        x = sepconv2d_bn(x, fliters, (3, 3))
        x = sepconv2d_bn(x, fliters, (3, 3), activation=None)
        x = MaxPooling2D((3, 3), padding='same', strides=2)(x)
        # 1X1 convolution layer를 activation 대신에 마지막에 적용시킨다.
        x = Add()([x, residual])

    ## Middle flow
    for i in range(8): 
        residual = x

        x = sepconv2d_bn(x, 1024, (3, 3))
        x = sepconv2d_bn(x, 1024, (3, 3))
        x = sepconv2d_bn(x, 1024, (3, 3), activation=None)

        x = Add()([x, residual])

    ## Exit flow
    residual = conv2d_bn(x, 1280, (1, 1), strides=2, activation=None)  

    x = Activation(activation='relu')(x)
    x = sepconv2d_bn(x, 1024, (3, 3))
    x = sepconv2d_bn(x, 1280, (3, 3), activation=None)  
    x = MaxPooling2D((3, 3), padding='same', strides=2)(x)  

    x = Add()([x, residual])

    x = sepconv2d_bn(x, 2048, (3, 3))
    x = sepconv2d_bn(x, 2560, (3, 3))
    x = cbam_block(x)
    x = GlobalAveragePooling2D()(x)
    
    #     x = Dense(2560)(x)
    #     x = BatchNormalization()(x)
    #     x = Activation(activation='relu')(x)
    
    model_output = Dense(classes, activation='softmax')(x)
    model = Model(model_input, model_output, name='Xception')
    return model

class LearningRateSchedule(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%2 == 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr*0.94)

input_shape = (160, 160, 3)

# X_train = X_train.astype('float32')/255.
# X_test = X_test.astype('float32')/255.

# y_train = to_categorical(y_train, num_classes=2)
# y_test = to_categorical(y_test, num_classes=2)

model_input = Input(shape=input_shape)

model = Xception(model_input,1)

model.summary()

class LearningRateSchedule(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%2 == 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr*0.94)


X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.

input_shape = (160, 160, 3)

model_input = Input(shape=input_shape)

model = Xception(model_input,1)

optimizer = SGD(lr=0.0009, momentum=0.9)
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=1e-5, amsgrad=False)

callbacks_list = [LearningRateSchedule()]

model.compile(optimizer, loss='categorical_crossentropy', metrics=['acc'])

model.fit(X_train, y_train, batch_size=64, epochs=40, validation_split=0.25, callbacks=callbacks_list)

# y_pred = model.predict(X_test)

# logloss = log_loss(y_test,y_pred)
# print(logloss)


# In[ ]:


# Ensenble 사용------------------------------------------------------

# 지정된 Layer의 Predict를 계산하기 위해, Layer 위치를 지정해 줌
first_output = model.layers[-3].output
model = tf.keras.models.Model(inputs=model.input, outputs=first_output)
a = model.predict(A)
print(a.shape)

b = []
for i in range(len(a)):
    b.append(a[i].flatten())
b = np.array(b)
print(b.shape)

c = B.reshape(-1,)


# In[ ]:


# Ex)10X10 Feature를 Flatten하기 위해 아래 함수 사용
def make_feature(data_set, feature_num):
    feature_size = a.shape[1]*a.shape[2]
    data_set = data_set[:,:feature_size*feature_num]
    data_set = data_set.flatten()
    data_set = data_set.reshape(-1,400)
    return data_set
X_train = make_feature(X_train,20)
X_val = make_feature(X_val,20)
X_test = make_feature(X_test,20)
y_train = list(y_train)
y_test = list(y_test)

def make_label(data_label,feature_num):
    d = []
    for i in range(len(data_label)):
        d.append(data_label[i]+[0]*feature_num)
    d = np.array(d)
    d = d.flatten()
    return d
y_train = make_label(y_train,20)
y_val = make_label(y_val,20)
y_test = make_label(y_test,20)

del b
del c
del A
del B


# In[ ]:


# RandomForest Model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 50, random_state = 0,max_features=1, bootstrap=False)

rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_test)
logloss = log_loss(y_test,y_pred)
print(logloss)


# In[ ]:


# Lightgbm Model
import lightgbm as lgb

train_ds = lgb.Dataset(X_train, label = y_train) 
val_ds = lgb.Dataset(X_val, label = y_val) 

params = {'learning_rate': 0.01,
          'max_depth': -1,
          'boosting': 'gbdt',
          'objective': 'binary',
          'metric': 'binary',
          'is_training_metric': True,
#           'feature_fraction': 1,
          'bagging_fraction': 0.7,
          'save_binary':True,
          'scale_pos_weight':1.2,
          'seed':2020}

model = lgb.train(params, train_ds, 3000, val_ds, verbose_eval=100, early_stopping_rounds=100)
y_pred = model.predict(X_test)
logloss = log_loss(y_test,y_pred)
print(logloss)

