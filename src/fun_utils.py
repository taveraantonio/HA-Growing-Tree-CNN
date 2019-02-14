# -*- coding: utf-8 -*-
"""
HA-Growing Tree-CNN - Source Code
Project 7.4 - Group05

Barbiero Pietro 
Sopegno Chiara
Tavera Antonio 

Created on Sun Aug  5 15:06:05 2018

"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from keras.applications.resnet50 import ResNet50
from keras.models import load_model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Activation
from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator


############################### LOAD DATASET ##########################################
# load the dataset from the directort
# 
# receive in input the path of the dataset(path)
# returns the data loaded
#######################################################################################
def load_dataset(input_path = '../data/DATASET_NPY/'):
	
	# load data
	data = []
	files = []
	error = False
	
	files.append(Path(input_path + 'x_train.npy'))
	files.append(Path(input_path + 'y_train.npy'))
	files.append(Path(input_path + 'x_validation.npy'))
	files.append(Path(input_path + 'y_validation.npy'))
	for file in files:
		if file.is_file():
			data.append(np.load(file))
		else:
			error = True
	
	if error:
		return None
	
	return data



############################### LOAD TEST SET #########################################
# load the test dataset from the directort
# 
# receive in input the path of the dataset(path)
# returns the data loaded
#######################################################################################
def load_test_set(input_path = '../data/DATASET_NPY/'):
	
	# load data
	data = []
	files = []
	error = False
	
	files.append(Path(input_path + 'x_test.npy'))
	files.append(Path(input_path + 'y_test.npy'))
	for file in files:
		if file.is_file():
			data.append(np.load(file))
		else:
			error = True
	
	if error:
		return None

	return data



############################### CREATE DATASET ########################################
# create the dtaset from the original one
# first load the orginal train set and test set, shuffle data, take the last 0.2 of 
# sample for validation, sort data according to label in ascemding order and save data
# to the output path.
# for the S-T-V dataset performs data augmentation 
# 
# receive in input the train percentage, the input and output path
#######################################################################################
def create_datasets(train_percentage=0.8, input_path = '../data/DATASET_NPY/', output_path = '../data/DATASET_NPY/'):
	
	#load H-AC data
	H_AC_x_train = np.load(input_path + 'H-AC/x_train.npy')
	H_AC_y_train = np.load(input_path + 'H-AC/y_train.npy')
	H_AC_x_test = np.load(input_path + 'H-AC/x_test.npy')
	H_AC_y_test = np.load(input_path + 'H-AC/y_test.npy')
	#shuffle data
	indeces = np.arange(0, len(H_AC_x_train))
	random.shuffle(indeces)
	H_AC_x_train = H_AC_x_train[indeces]
	H_AC_y_train = H_AC_y_train[indeces]
	#take last 0.2 of samples for validation
	train_size = int(np.round(train_percentage * len(H_AC_x_train)))
    #CONTROLLA POSSIBILE ERRORE CON I DUE PUNTI
    #############
	H_AC_x_validation = H_AC_x_train[train_size:]
	H_AC_y_validation = H_AC_y_train[train_size:]
	H_AC_x_train = H_AC_x_train[:train_size]
	H_AC_y_train = H_AC_y_train[:train_size]
	#sort data according to labels in ascending order
	indeces = np.argsort(H_AC_y_train)
	H_AC_x_train = H_AC_x_train[indeces]
	H_AC_y_train = H_AC_y_train[indeces]
	indeces = np.argsort(H_AC_y_validation)
	H_AC_x_validation = H_AC_x_validation[indeces]
	H_AC_y_validation = H_AC_y_validation[indeces]
	indeces = np.argsort(H_AC_y_test)
	H_AC_x_test = H_AC_x_test[indeces]
	H_AC_y_test = H_AC_y_test[indeces]
	#save a subset of original data
	os.makedirs(output_path + 'AC_H')
	np.save(output_path + 'AC_H/x_train', H_AC_x_train)
	np.save(output_path + 'AC_H/x_validation', H_AC_x_validation)
	np.save(output_path + 'AC_H/x_test', H_AC_x_test)
	np.save(output_path + 'AC_H/y_train', H_AC_y_train)
	np.save(output_path + 'AC_H/y_validation', H_AC_y_validation)
	np.save(output_path + 'AC_H/y_test', H_AC_y_test)
	
	#load H-AD data
	H_AD_x_train = np.load(input_path + 'H-AD/x_train.npy')
	H_AD_y_train = np.load(input_path + 'H-AD/y_train.npy')
	H_AD_x_test = np.load(input_path + 'H-AD/x_test.npy')
	H_AD_y_test = np.load(input_path + 'H-AD/y_test.npy')
	# shuffle data
	indeces = np.arange(0, len(H_AD_x_train))
	random.shuffle(indeces)
	H_AD_x_train = H_AD_x_train[indeces]
	H_AD_y_train = H_AD_y_train[indeces]
	# take last 0.2 of samples for validation
	train_size = int(np.round(train_percentage * len(H_AD_x_train)))
	H_AD_x_validation = H_AD_x_train[train_size:]
	H_AD_y_validation = H_AD_y_train[train_size:]
	H_AD_x_train = H_AD_x_train[:train_size]
	H_AD_y_train = H_AD_y_train[:train_size]
	#sort data according to labels in ascending order
	indeces = np.argsort(H_AD_y_train)
	H_AD_x_train = H_AD_x_train[indeces]
	H_AD_y_train = H_AD_y_train[indeces]
	indeces = np.argsort(H_AD_y_validation)
	H_AD_x_validation = H_AD_x_validation[indeces]
	H_AD_y_validation = H_AD_y_validation[indeces]
	indeces = np.argsort(H_AD_y_test)
	H_AD_x_test = H_AD_x_test[indeces]
	H_AD_y_test = H_AD_y_test[indeces]
	#save a subset of original data
	os.makedirs(output_path + 'AD_HH')
	np.save(output_path + 'AD_HH/x_train', H_AD_x_train)
	np.save(output_path + 'AD_HH/x_validation', H_AD_x_validation)
	np.save(output_path + 'AD_HH/x_test', H_AD_x_test)
	np.save(output_path + 'AD_HH/y_train', H_AD_y_train)
	np.save(output_path + 'AD_HH/y_validation', H_AD_y_validation)
	np.save(output_path + 'AD_HH/y_test', H_AD_y_test)

	
	#load S-T-V data
	S_T_V_x_train = np.load(input_path + 'S-T-V/x_train.npy')
	S_T_V_y_train = np.load(input_path + 'S-T-V/y_train.npy')
	S_T_V_x_test = np.load(input_path + 'S-T-V/x_test.npy')
	S_T_V_y_test = np.load(input_path + 'S-T-V/y_test.npy')
		
	#shuffle data
	indeces = np.arange(0, len(S_T_V_y_train))
	random.shuffle(indeces)
	S_T_V_x_train = S_T_V_x_train[indeces]
	S_T_V_y_train = S_T_V_y_train[indeces]
	# take last 0.2 of samples for validation
	train_size = int(np.round(train_percentage * len(S_T_V_y_train)))
	S_T_V_x_validation = S_T_V_x_train[train_size:]
	S_T_V_y_validation = S_T_V_y_train[train_size:]
	S_T_V_x_train = S_T_V_x_train[:train_size]
	S_T_V_y_train = S_T_V_y_train[:train_size]
	
#	######################### DATA AUGMENTATION ###############################
#	
#	#setting parameters for data augmentation for S T V classes 
#	datagen = ImageDataGenerator(	
#			featurewise_center=True,
#			featurewise_std_normalization=True,
#			rotation_range=30,
#			width_shift_range=0.2,
#			height_shift_range=0.2,
#			shear_range= 15,
#			channel_shift_range = 0.005,
#			zoom_range=0.2,
#			horizontal_flip = True,
#			vertical_flip = True)
#	
#	datagen.fit(S_T_V_x_train)
#	size = len(S_T_V_y_train)
#	#starting data augmentation loop on train set
#	i = 0
#	new_data = datagen.flow(S_T_V_x_train, S_T_V_y_train, batch_size=1, shuffle=False)
#	for x_train, y_train in new_data:
#		x_train = x_train.reshape(1, 224, 224, 3)
#		x_train = np.array(x_train, np.uint8)
#		S_T_V_x_train = np.concatenate((S_T_V_x_train, x_train))
#		S_T_V_y_train = np.concatenate((S_T_V_y_train, y_train))
#		i+=1
#		if i == size:
#			break
#		
#	datagen.fit(S_T_V_x_validation)
#	size = len(S_T_V_y_validation)
#	#starting data augmentation loop on validation set 
#	i = 0
#	new_data = datagen.flow(S_T_V_x_validation, S_T_V_y_validation, batch_size=1, shuffle=False)
#	for x_validation, y_validation in new_data:
#		x_validation = x_train.reshape(1, 224, 224, 3)
#		x_validation = np.array(x_train, np.uint8)
#		S_T_V_x_validation = np.concatenate((S_T_V_x_validation, x_validation))
#		S_T_V_y_validation = np.concatenate((S_T_V_y_validation, y_validation))
#		i+=1
#	
#		if i == size:
#			break
#		
#	####################### END DATA AUGMENTATION ###############################
		
	# sort data according to labels in ascending order
	indeces = np.argsort(S_T_V_y_train)
	S_T_V_x_train = S_T_V_x_train[indeces]
	S_T_V_y_train = S_T_V_y_train[indeces]
	indeces = np.argsort(S_T_V_y_validation)
	S_T_V_x_validation = S_T_V_x_validation[indeces]
	S_T_V_y_validation = S_T_V_y_validation[indeces]
	indeces = np.argsort(S_T_V_y_test)
	S_T_V_x_test = S_T_V_x_test[indeces]
	S_T_V_y_test = S_T_V_y_test[indeces]
	
	# save a subset of original data
	os.makedirs(output_path + 'S_T_V')
	np.save(output_path + 'S_T_V/x_train', S_T_V_x_train)
	np.save(output_path + 'S_T_V/x_validation', S_T_V_x_validation)
	np.save(output_path + 'S_T_V/x_test', S_T_V_x_test)
	np.save(output_path + 'S_T_V/y_train', S_T_V_y_train)
	np.save(output_path + 'S_T_V/y_validation', S_T_V_y_validation)
	np.save(output_path + 'S_T_V/y_test', S_T_V_y_test)
	
#	########################## TEST S/TV #############################
#	S_x_train = S_T_V_x_train[S_T_V_y_train==0]
#	S_y_train = S_T_V_y_train[S_T_V_y_train==0]
#	T_V_x_train = S_T_V_x_train[S_T_V_y_train!=0]
#	T_V_y_train = S_T_V_y_train[S_T_V_y_train!=0]	
#	T_V_y_train -= 1
#	
#	S_x_validation = S_T_V_x_validation[S_T_V_y_validation==0]
#	S_y_validation = S_T_V_y_validation[S_T_V_y_validation==0]
#	T_V_x_validation = S_T_V_x_validation[S_T_V_y_validation!=0]
#	T_V_y_validation = S_T_V_y_validation[S_T_V_y_validation!=0] 
#	T_V_y_validation -= 1
#	
#	S_x_test = S_T_V_x_test[S_T_V_y_test==0]
#	S_y_test = S_T_V_y_test[S_T_V_y_test==0]
#	T_V_x_test = S_T_V_x_test[S_T_V_y_test!=0]
#	T_V_y_test = S_T_V_y_test[S_T_V_y_test!=0] 
#	T_V_y_test -= 1
#	
#	#save a subset of original data
#	os.makedirs(output_path + 'S')
#	np.save(output_path + 'S/x_train', S_x_train)
#	np.save(output_path + 'S/x_validation', S_x_validation)
#	np.save(output_path + 'S/x_test', S_x_test)
#	np.save(output_path + 'S/y_train', S_y_train)
#	np.save(output_path + 'S/y_validation', S_y_validation)
#	np.save(output_path + 'S/y_test', S_y_test)
#	
#	#save a subset of original data
#	os.makedirs(output_path + 'T_V')
#	np.save(output_path + 'T_V/x_train', T_V_x_train)
#	np.save(output_path + 'T_V/x_validation', T_V_x_validation)
#	np.save(output_path + 'T_V/x_test', T_V_x_test)
#	np.save(output_path + 'T_V/y_train', T_V_y_train)
#	np.save(output_path + 'T_V/y_validation', T_V_y_validation)
#	np.save(output_path + 'T_V/y_test', T_V_y_test)
#		
#	############################# END TEST ###########################
	
	return
	


############################### CREATE NEURAL NETWORK #################################
# create the neural network for the node
# load Resnet if already saved or save it, create a classifier model composed of 
# ResNet50 + a dense network, save the model and returns it
# 
# receive in input the name of the model, the lenght and height of the image and the 
# maximum number of output for that model 
# returns the model
#######################################################################################
def create_neural_network(model_name, img_l, img_h, max_output):
	
	#load ResNet if already exists 
	#download and save it if not
	try:
		res_net = load_model('ResNet50.h5')
	except:
		res_net = ResNet50(weights='imagenet', include_top=False, input_shape=(img_l, img_h, 3))
		res_net.compile(loss='categorical_crossentropy',
				optimizer=optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
				metrics=['acc'])
		res_net.save('ResNet50.h5')

	#res_net.summary()
	#freeze the first 45 layers of the ResNet except the last 5 
	for layer in res_net.layers[:-5]:
	    layer.trainable = False
	
	#build a classifier model composed of ResNet50 + a dense network
	#the sequential model is a linear stack of layers
	model = Sequential()
	#add the ResNet50 convolutional base model
	model.add(res_net)
	#add new layers
	#add droput layer, consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
	model.add(Dropout(0.8))
	#add Flatten layer, to flatten the input 
	model.add(Flatten())
	#add another droput layer
	model.add(Dropout(0.8))
	#add a dense layer (fully connected layer) with 1024 neurons with ReLU as activation function
	model.add(Dense(1024, kernel_initializer='random_uniform', activation='relu'))
	model.add(Dropout(0.8))
	#add another dense layer with a maximum nuber = num of classes 
	model.add(Dense(max_output, kernel_initializer='random_uniform', name='before_softmax'))
	#add the softmax, not done in the definition before in order to take values before softmax in likelihood computation
	model.add(Activation('softmax'))
	
	#model.summary()
	model.save(model_name)
	
	return model



######################### CREATE SMALL NEURAL NETWORK #################################
# create a small neural network for debug purpouse, no use of ResNet here
# 
# receive in input the name of the model, the lenght and height of the image and the 
# maximum number of output for that model 
# returns the model
#######################################################################################
def create_neural_network_small(model_name, img_l, img_h, max_output):
	input_shape = tuple((img_l, img_h, 3))
	
	model = Sequential()
	#add a convolutional layer
	model.add(Conv2D(4, kernel_size=(3, 3), strides=(1, 1),
	                 activation='relu',
	                 input_shape=input_shape))
	#add a pooling layer
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	#add a flatten layer to flattens the input
	model.add(Flatten())
	#add a fully connected layer
	model.add(Dense(6, activation='relu'))
	model.add(Dense(max_output, name='before_softmax'))
	#add softmax
	model.add(Activation('softmax'))
	
	model.summary()
	model.save(model_name)
	
	return model



############################## TRAIN MODEL ############################################
# train the received model with the new label
# if the number of label received (number of class the model have to discriminate) is 1 
# compile the model with the binary crossentropy loss function otherwise with the
# categorical one. For each of them uses the Adam optimizer.   
# Perform early stopping to avoid overfitting and plot train accuracy 
# 
# 
# Receive as input the non trained model, the train and validation set, the train and 
# validation model, the model name, the learning rate, the maximum number of epochs 
# the batch size, the verbose and the validation split. It is also to avoi plotting 
# results by setting plot to False
# Returns the trained model 
#######################################################################################
def train_model(model, X, y, X_val, y_val, model_name, lr=1e-4, epochs=50, batch_size=10, verbose=1, validation_split=0.25, plot=True):

	#Compile the model
	if len(np.unique(y)) == 1:
		model.compile(loss='binary_crossentropy',
				#optimizer=optimizers.RMSprop(lr=1e-4),
				optimizer=optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
				#optimizer=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0),
				metrics=['acc'])
	else:
		model.compile(loss='categorical_crossentropy',
				#optimizer=optimizers.RMSprop(lr=1e-4),
				optimizer=optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
				#optimizer=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0),
				metrics=['acc'])
		
	#one-hot encoding for the train set labels (represent the label as binary vector)
	onehot_encoder = OneHotEncoder(sparse=False)
	y = y.reshape(len(y), 1)
	y_encoded = onehot_encoder.fit_transform(y)
	
	#one-hot encoding for the validation set labels 
	y_val = y_val.reshape(len(y_val), 1)
	y_val_encoded = onehot_encoder.fit_transform(y_val)
	
	#define early stopping callback
	earlystop = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=5, \
                          verbose=1, mode='auto')	
	callbacks_list = [earlystop]

	#train the model
	history = model.fit(X, y_encoded, epochs=epochs, batch_size=batch_size, 
					 verbose=verbose, validation_data=(X_val, y_val_encoded),
					 callbacks=callbacks_list)
	#save the trained model
	model.save(model_name)
	
	#plot results
	if plot == True:
		acc = history.history['acc']
		val_acc = history.history['val_acc']
		loss = history.history['loss']
		val_loss = history.history['val_loss']
		
		epochs = range(len(acc))
		
		plt.figure()
		plt.plot(epochs, acc, 'b', label='Training acc')
		plt.plot(epochs, val_acc, 'r', label='Validation acc')
		plt.title('Training and validation accuracy')
		plt.legend()
		#plt.savefig('accuracy_%s.png' %(np.unique(y)), dpi=500)
		plt.show()
		
		plt.figure()
		plt.plot(epochs, loss, 'b', label='Training loss')
		plt.plot(epochs, val_loss, 'r', label='Validation loss')
		plt.title('Training and validation loss')
		plt.legend()
		#plt.savefig('loss_%s.png' %(np.unique(y)), dpi=500)
		plt.show()
	
	#return the trained model 
	return model



############################## TEST MODEL ############################################
# test the trained model and return the accuracy 
# 
# Receive as input the trained model, the train set and its labels 
# Returns the accuracy
#######################################################################################
def test_model(model, X, y):
	#make a prediction on the test set
	test_predictions = model.predict(X)
	test_predictions = np.round(test_predictions)
	
	#report the accuracy
	accuracy = accuracy_score(y, test_predictions)
	print("Accuracy: " + str(accuracy))
	
	return accuracy
