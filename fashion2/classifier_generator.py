#Written by: Rajeev Sahay

# The purpose of this code is to measure the time taken to reduce the MNIST fashion dataset
# to various compression levels and train a classifier with that corresponing input layer size

import numpy as np
from keras.models import Sequential
import keras
from keras.datasets import fashion_mnist
from keras.layers import Dense
from keras.models import load_model
from keras import backend as K
import time


#Import dataset and normalize to [0,1]
(data_train, labels_train), (data_test, labels_test) = fashion_mnist.load_data()
data_train = data_train/255.0
data_test = data_test/255.0

#Flatten dataset (New shape for training and testing set is (60000,784) and (10000, 784))
data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))

#Create labels as one-hot vectors
labels_train = keras.utils.np_utils.to_categorical(labels_train, num_classes=10)
labels_test = keras.utils.np_utils.to_categorical(labels_test, num_classes=10)


#Create the model
def fc_model_16():

    model = Sequential()
    model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=16))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model

def fc_model_32():

    model = Sequential()
    model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=32))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model

def fc_model_47():

    model = Sequential()
    model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=47))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model

def fc_model_64():

    model = Sequential()
    model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=64))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model

def fc_model_80():

    model = Sequential()
    model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=80))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model

def fc_model_94():

    model = Sequential()
    model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=94))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model

def fc_model_157():

    model = Sequential()
    model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=157))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model

def fc_model_784():

    model = Sequential()
    model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=784))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model


#Create matrix to store times in
times = np.zeros([8, 2])
times[0, 0] = 16
times[1, 0] = 32
times[2, 0] = 47
times[3, 0] = 64
times[4, 0] = 80
times[5, 0] = 94
times[6, 0] = 157
times[7, 0] = 784


start_time_16 = time.time()
model_16 = fc_model_16()
#Load pre-processing autoencoder
pp_ae_eps2550_16 = load_model('daes/dae_16_eps2550.h5')
#Obtain hidden layer representation of data from DAE
get_hidden_layer_output_16 = K.function([pp_ae_eps2550_16.layers[0].input], [pp_ae_eps2550_16.layers[2].output])
#Get hidden layer representation of DAE of attacked data for k=16
data_train_16 = get_hidden_layer_output_16([data_train])[0]
data_test_16 = get_hidden_layer_output_16([data_test])[0]
#Compile model using cross entropy as loss and adam as optimizer
model_16.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
model_16.fit(data_train_16, labels_train, validation_data=(data_test_16, labels_test), epochs=100, batch_size=200, shuffle=True)
end_time_16 = time.time() - start_time_16
times[0, 1] = end_time_16
#Save the model
model_16.save('classifiers/fc-16-100-100-10-2550.h5')



start_time_32 = time.time()
model_32 = fc_model_32()
#Load pre-processing autoencoder
pp_ae_eps2550_32 = load_model('daes/dae_32_eps2550.h5')
#Obtain hidden layer representation of data from DAE
get_hidden_layer_output_32 = K.function([pp_ae_eps2550_32.layers[0].input], [pp_ae_eps2550_32.layers[2].output])
#Get hidden layer representation of DAE of attacked data for k=16
data_train_32 = get_hidden_layer_output_32([data_train])[0]
data_test_32 = get_hidden_layer_output_32([data_test])[0]
#Compile model using cross entropy as loss and adam as optimizer
model_32.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
model_32.fit(data_train_32, labels_train, validation_data=(data_test_32, labels_test), epochs=100, batch_size=200, shuffle=True)
end_time_32 = time.time() - start_time_32
times[1, 1] = end_time_32
#Save the model
model_32.save('classifiers/fc-32-100-100-10-2550.h5')



start_time_47 = time.time()
model_47 = fc_model_47()
#Load pre-processing autoencoder
pp_ae_eps2550_47 = load_model('daes/dae_47_eps2550.h5')
#Obtain hidden layer representation of data from DAE
get_hidden_layer_output_47 = K.function([pp_ae_eps2550_47.layers[0].input], [pp_ae_eps2550_47.layers[2].output])
#Get hidden layer representation of DAE of attacked data for k=16
data_train_47 = get_hidden_layer_output_47([data_train])[0]
data_test_47 = get_hidden_layer_output_47([data_test])[0]
#Compile model using cross entropy as loss and adam as optimizer
model_47.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
model_47.fit(data_train_47, labels_train, validation_data=(data_test_47, labels_test), epochs=100, batch_size=200, shuffle=True)
end_time_47 = time.time() - start_time_47
times[2, 1] = end_time_47
#Save the model
model_47.save('classifiers/fc-47-100-100-10-2550.h5')



start_time_64 = time.time()
model_64 = fc_model_64()
#Load pre-processing autoencoder
pp_ae_eps2550_64 = load_model('daes/dae_64_eps2550.h5')
#Obtain hidden layer representation of data from DAE
get_hidden_layer_output_64 = K.function([pp_ae_eps2550_64.layers[0].input], [pp_ae_eps2550_64.layers[2].output])
#Get hidden layer representation of DAE of attacked data for k=16
data_train_64 = get_hidden_layer_output_64([data_train])[0]
data_test_64 = get_hidden_layer_output_64([data_test])[0]
#Compile model using cross entropy as loss and adam as optimizer
model_64.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
model_64.fit(data_train_64, labels_train, validation_data=(data_test_64, labels_test), epochs=100, batch_size=200, shuffle=True)
end_time_64 = time.time() - start_time_64
times[3, 1] = end_time_64
#Save the model
model_64.save('classifiers/fc-64-100-100-10-2550.h5')



start_time_80 = time.time()
model_80 = fc_model_80()
#Load pre-processing autoencoder
pp_ae_eps2550_80 = load_model('daes/dae_80_eps2550.h5')
#Obtain hidden layer representation of data from DAE
get_hidden_layer_output_80 = K.function([pp_ae_eps2550_80.layers[0].input], [pp_ae_eps2550_80.layers[2].output])
#Get hidden layer representation of DAE of attacked data for k=16
data_train_80 = get_hidden_layer_output_80([data_train])[0]
data_test_80 = get_hidden_layer_output_80([data_test])[0]
#Compile model using cross entropy as loss and adam as optimizer
model_80.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
model_80.fit(data_train_80, labels_train, validation_data=(data_test_80, labels_test), epochs=100, batch_size=200, shuffle=True)
end_time_80 = time.time() - start_time_80
times[4, 1] = end_time_80
#Save the model
model_80.save('classifiers/fc-80-100-100-10-2550.h5')



start_time_94 = time.time()
model_94 = fc_model_94()
#Load pre-processing autoencoder
pp_ae_eps2550_94 = load_model('daes/dae_94_eps2550.h5')
#Obtain hidden layer representation of data from DAE
get_hidden_layer_output_94 = K.function([pp_ae_eps2550_94.layers[0].input], [pp_ae_eps2550_94.layers[2].output])
#Get hidden layer representation of DAE of attacked data for k=16
data_train_94 = get_hidden_layer_output_94([data_train])[0]
data_test_94 = get_hidden_layer_output_94([data_test])[0]
#Compile model using cross entropy as loss and adam as optimizer
model_94.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
model_94.fit(data_train_94, labels_train, validation_data=(data_test_94, labels_test), epochs=100, batch_size=200, shuffle=True)
end_time_94 = time.time() - start_time_94
times[5, 1] = end_time_94
#Save the model
model_94.save('classifiers/fc-94-100-100-10-2550.h5')



start_time_157 = time.time()
model_157 = fc_model_157()
#Load pre-processing autoencoder
pp_ae_eps2550_157 = load_model('daes/dae_157_eps2550.h5')
#Obtain hidden layer representation of data from DAE
get_hidden_layer_output_157 = K.function([pp_ae_eps2550_157.layers[0].input], [pp_ae_eps2550_157.layers[2].output])
#Get hidden layer representation of DAE of attacked data for k=16
data_train_157 = get_hidden_layer_output_157([data_train])[0]
data_test_157 = get_hidden_layer_output_157([data_test])[0]
#Compile model using cross entropy as loss and adam as optimizer
model_157.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
model_157.fit(data_train_157, labels_train, validation_data=(data_test_157, labels_test), epochs=100, batch_size=200, shuffle=True)
end_time_157 = time.time() - start_time_157
times[6, 1] = end_time_157
#Save the model
model_157.save('classifiers/fc-157-100-100-10-2550.h5')



start_time_784 = time.time()
model_784 = fc_model_784()
#Compile model using cross entropy as loss and adam as optimizer
model_784.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
model_784.fit(data_train, labels_train, validation_data=(data_test, labels_test), epochs=100, batch_size=200, shuffle=True)
end_time_784 = time.time() - start_time_784
times[7, 1] = end_time_784



np.savetxt("times.csv", times, delimiter=",")
