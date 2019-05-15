#Written by: Rajeev Sahay

# This code trains and saves a denoising autoencoder (DAE) using the MNIST hand written digit dataset,
# perterbed values of half of the same dataset according to the Fast Gradient Sign attack and an attack magnitude
# of 0.25, and perterbed values of the other half of the same dataset according to the Fast Gradient Sign attack
# and an attack magnitude of 0.50


import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import load_model
from keras import backend
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper


#Load MNIST data and normalize to [0,1]
(data_train, _), (data_test, _) = mnist.load_data()
data_train = data_train/255.0
data_test = data_test/255.0

#Flatten dataset (New shape for training and testing set is (60000,784) and (10000, 784))
data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))


#Load target classifier model whose gradients will be used to create adversarial examples
keras_model = load_model('classifiers/fc-784-100-100-10-defender-model.h5')
backend.set_learning_phase(False)

data_train_shuffle = data_train
data_train1 = data_train_shuffle[0:30000,0:784]
data_train2 = data_train_shuffle[30000:60000,0:784]


#Create adversarial examples on testing data
sess =  backend.get_session()
eta1 = 0.25
eta2 = 0.50
wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
adv_train_x1 = fgsm.generate_np(data_train1, eps=eta1, clip_min=0., clip_max=1.)
adv_train_x2 = fgsm.generate_np(data_train2, eps=eta2, clip_min=0., clip_max=1.)
adv_train_x = np.vstack([adv_train_x1, adv_train_x2])
adv_test_x = fgsm.generate_np(data_test, eps=eta1, clip_min=0., clip_max=1.)

#Total datasets
data_total_train = np.vstack([data_train, adv_train_x])
data_total_test = np.vstack([data_test, adv_test_x])

#Create labels that correspond to clean reconstructions
labels_total_train = np.vstack([data_train, data_train])
labels_total_test = np.vstack([data_test, data_test])

#Create the model
def autoencoder_16():

    model = Sequential()
    model.add(Dense(512, activation=None, use_bias=True, kernel_initializer="uniform", input_dim=784))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(16, activation=None, kernel_initializer="uniform"))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(512, activation=None, kernel_initializer="uniform"))
    model.add(Dense(784, activation="sigmoid", kernel_initializer="uniform"))
    return model

def autoencoder_32():

    model = Sequential()
    model.add(Dense(512, activation=None, use_bias=True, kernel_initializer="uniform", input_dim=784))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(32, activation=None, kernel_initializer="uniform"))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(512, activation=None, kernel_initializer="uniform"))
    model.add(Dense(784, activation="sigmoid", kernel_initializer="uniform"))
    return model

def autoencoder_47():

    model = Sequential()
    model.add(Dense(512, activation=None, use_bias=True, kernel_initializer="uniform", input_dim=784))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(47, activation=None, kernel_initializer="uniform"))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(512, activation=None, kernel_initializer="uniform"))
    model.add(Dense(784, activation="sigmoid", kernel_initializer="uniform"))
    return model

def autoencoder_64():

    model = Sequential()
    model.add(Dense(512, activation=None, use_bias=True, kernel_initializer="uniform", input_dim=784))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(64, activation=None, kernel_initializer="uniform"))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(512, activation=None, kernel_initializer="uniform"))
    model.add(Dense(784, activation="sigmoid", kernel_initializer="uniform"))
    return model

def autoencoder_80():

    model = Sequential()
    model.add(Dense(512, activation=None, use_bias=True, kernel_initializer="uniform", input_dim=784))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(80, activation=None, kernel_initializer="uniform"))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(512, activation=None, kernel_initializer="uniform"))
    model.add(Dense(784, activation="sigmoid", kernel_initializer="uniform"))
    return model

def autoencoder_94():

    model = Sequential()
    model.add(Dense(512, activation=None, use_bias=True, kernel_initializer="uniform", input_dim=784))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(94, activation=None, kernel_initializer="uniform"))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(512, activation=None, kernel_initializer="uniform"))
    model.add(Dense(784, activation="sigmoid", kernel_initializer="uniform"))
    return model

def autoencoder_157():

    model = Sequential()
    model.add(Dense(512, activation=None, use_bias=True, kernel_initializer="uniform", input_dim=784))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(157, activation=None, kernel_initializer="uniform"))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(512, activation=None, kernel_initializer="uniform"))
    model.add(Dense(784, activation="sigmoid", kernel_initializer="uniform"))
    return model



model_16 = autoencoder_16()
#Compile model using mean squared error as loss and adam as optimizer
model_16.compile(loss='mean_squared_error', optimizer='adam')
#Train model using input of clean and corrupted data and fit to clean reconstructions only
model_16.fit(data_total_train, labels_total_train, validation_data=(data_total_test, labels_total_test), epochs=150, batch_size=256, shuffle=True)
#Save the model
model_16.save('daes/dae_16_eps2550.h5')


model_32 = autoencoder_32()
#Compile model using mean squared error as loss and adam as optimizer
model_32.compile(loss='mean_squared_error', optimizer='adam')
#Train model using input of clean and corrupted data and fit to clean reconstructions only
model_32.fit(data_total_train, labels_total_train, validation_data=(data_total_test, labels_total_test), epochs=150, batch_size=256, shuffle=True)
#Save the model
model_32.save('daes/dae_32_eps2550.h5')


model_47 = autoencoder_47()
#Compile model using mean squared error as loss and adam as optimizer
model_47.compile(loss='mean_squared_error', optimizer='adam')
#Train model using input of clean and corrupted data and fit to clean reconstructions only
model_47.fit(data_total_train, labels_total_train, validation_data=(data_total_test, labels_total_test), epochs=150, batch_size=256, shuffle=True)
#Save the model
model_47.save('daes/dae_47_eps2550.h5')


model_64 = autoencoder_64()
#Compile model using mean squared error as loss and adam as optimizer
model_64.compile(loss='mean_squared_error', optimizer='adam')
#Train model using input of clean and corrupted data and fit to clean reconstructions only
model_64.fit(data_total_train, labels_total_train, validation_data=(data_total_test, labels_total_test), epochs=150, batch_size=256, shuffle=True)
#Save the model
model_64.save('daes/dae_64_eps2550.h5')


model_80 = autoencoder_80()
#Compile model using mean squared error as loss and adam as optimizer
model_80.compile(loss='mean_squared_error', optimizer='adam')
#Train model using input of clean and corrupted data and fit to clean reconstructions only
model_80.fit(data_total_train, labels_total_train, validation_data=(data_total_test, labels_total_test), epochs=150, batch_size=256, shuffle=True)
#Save the model
model_80.save('daes/dae_80_eps2550.h5')


model_94 = autoencoder_94()
#Compile model using mean squared error as loss and adam as optimizer
model_94.compile(loss='mean_squared_error', optimizer='adam')
#Train model using input of clean and corrupted data and fit to clean reconstructions only
model_94.fit(data_total_train, labels_total_train, validation_data=(data_total_test, labels_total_test), epochs=150, batch_size=256, shuffle=True)
#Save the model
model_94.save('daes/dae_94_eps2550.h5')

model_157 = autoencoder_157()
#Compile model using mean squared error as loss and adam as optimizer
model_157.compile(loss='mean_squared_error', optimizer='adam')
#Train model using input of clean and corrupted data and fit to clean reconstructions only
model_157.fit(data_total_train, labels_total_train, validation_data=(data_total_test, labels_total_test), epochs=150, batch_size=256, shuffle=True)
#Save the model
model_157.save('daes/dae_157_eps2550.h5')
