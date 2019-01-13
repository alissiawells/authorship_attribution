# -*- coding:utf-8 -*-
# CNN multi-class classifier based on tf + doc2vec from gensim
from __future__ import print_function
import keras
import re
import numpy as np
from numpy import *
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import Infertest
import Clean
import Genclf

author_num = 50
text_num = 1000
feature_dim = 300
classification_dim = int(0.03*feature_dim)
ker_num = 16
ker_size = 3
train_epoch = 20
test_size = author_num * text_num / 5
train_size = 4 * test_size

vector_list_train = []
author_list_train = []
t_train = []
vector_list_test = []
author_list_test = []

# training dataset: docvec,authorID = (vector_list_train,author_list_train) 
Infertest.infertest()
Clean.clean()
Genclf.genclf()

doc_train = open('clf_train.txt') 
lines = doc_train.readlines()
for l in lines:
	m = re.match('([^\s]+) (.+)', l)
	author = m.group(1)
	author_list_train.append(author)
	vector = m.group(2)
	vector = re.split(' ',vector)
	float_vector=[]
	for str in vector:
		k = re.match('(\d+)\:(.+)', str)
		strval = k.group(2)
		float_vector.append(float(strval))	
	vector_list_train.append(float_vector)
doc_train.close()

# test dataset: docvec,authorID = (vector_list_test,author_list_test) 
doc_test = open('clf_test.txt') 
lines = doc_test.readlines()
for l in lines:
	m = re.match('([^\s]+) (.+)', l)
	author = m.group(1)
	author_list_test.append(author)
	vector = m.group(2)
	vector = re.split(' ',vector)
	float_vector=[]
	for str in vector:
		k = re.match('(\d+)\:(.+)', str)
		strval = k.group(2)
		float_vector.append(float(strval))	
	vector_list_test.append(float_vector)
doc_test.close()

X_train = np.array(vector_list_train)
Y_train = np.array(author_list_train)
X_test = np.array(vector_list_test)
Y_test = np.array(author_list_test)

Y_train = np_utils.to_categorical(Y_train, author_num)
Y_test = np_utils.to_categorical(Y_test, author_num)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

# 1 convolution and pooling layer + 1 fully connected layer
model = Sequential()
model.add(Conv1D(ker_num, ker_size, activation='relu',border_mode='same', #input_shape=(classification_dim,1)))
input_shape=(feature_dim,1)))
model.add(MaxPooling1D(2))

model.add(Dropout(0.5))									# prevent overfitting
model.add(Flatten())									# transform multidimension to 1 dimension, usually used between conv. and densely connencted 
model.add(Dense(author_num, activation='softmax'))   	# dense is densly connected layer
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

# train and test the model
model.fit(X_train, Y_train, epochs=train_epoch, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)

print('Experiment:',author_num, "*", text_num)
print('Kernel number:',ker_num)
print('Kernel size:',ker_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
