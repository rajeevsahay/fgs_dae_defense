#Written by: Rajeev Sahay

# This code trains and saves a denoising autoencoder (DAE) using the MNIST fashion dataset,
# perterbed values of half of the same dataset according to the Fast Gradient Sign attack and an attack magnitude
# of 0.25, and perterbed values of the other half of the same dataset according to the Fast Gradient Sign attack
# and an attack magnitude of 0.50

import numpy as np
import keras
from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.layers import Dense
from keras.models import load_model
from keras import backend
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from sklearn.decomposition import PCA
from keras import backend as K

#Load MNIST data and normalize to [0,1]
(data_train, labels_train), (data_test, labels_test) = fashion_mnist.load_data()
data_train = data_train/255.0
data_test = data_test/255.0

#Flatten dataset (New shape for training and testing set is (60000,784) and (10000, 784))
data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))

#Create labels as one-hot vectors
labels_train = keras.utils.np_utils.to_categorical(labels_train, num_classes=10)
labels_test = keras.utils.np_utils.to_categorical(labels_test, num_classes=10)


#Load classifier model whose gradients will be used to create adversarial examples
keras_model = load_model('classifiers/fc-784-100-100-10-defender-model.h5')
atkr_clfr = load_model('classifiers/fc-784-200-200-100-10-attacker-model.h5')
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

#Create the models

def trad_ae():

    model = Sequential()
    model.add(Dense(512, activation=None, use_bias=True, kernel_initializer="uniform", input_dim=784))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(47, activation=None, kernel_initializer="uniform"))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(512, activation=None, kernel_initializer="uniform"))
    model.add(Dense(784, activation="sigmoid", kernel_initializer="uniform"))
    return model


def fc_model_47():

    model = Sequential()
    model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=47))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model


#Train traditional AE
trad_ae = trad_ae()
trad_ae.compile(loss='mean_squared_error', optimizer='adam')
trad_ae.fit(data_train, data_train, validation_data=(data_test, data_test), epochs=150, batch_size=256, shuffle=True)
trad_ae.save('daes/trad_ae.h5')

#Train classifier with traditional AE input
fc_model_trad_ae = fc_model_47()
hidden_layer_trad_ae = K.function([trad_ae.layers[0].input], [trad_ae.layers[2].output])
data_train_trad_ae = hidden_layer_trad_ae([data_train])[0]
data_test_trad_ae = hidden_layer_trad_ae([data_test])[0]
fc_model_trad_ae.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fc_model_trad_ae.fit(data_train_trad_ae, labels_train, validation_data=(data_test_trad_ae, labels_test), epochs=100, batch_size=200, shuffle=True)
fc_model_trad_ae.save('classifiers/fc-47-100-100-10-trad-ae.h5')


#Adversarially (re)train defender model
keras_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
keras_model.fit(data_total_train, np.vstack([labels_train, labels_train]), validation_data=(data_total_test, np.vstack([labels_test, labels_test])), epochs=100, batch_size=200, shuffle=True)
#Save the model
keras_model.save('classifiers/fc-784-100-100-10-adv-trn.h5')

#Adversarially (re)train attacker model
atkr_clfr.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
atkr_clfr.fit(data_total_train, np.vstack([labels_train, labels_train]), validation_data=(data_total_test, np.vstack([labels_test, labels_test])), epochs=100, batch_size=200, shuffle=True)
#Save the model
atkr_clfr.save('classifiers/fc-784-200-200-100-10-adv-trn-atkr.h5')

#PCA model
pca = PCA(47)
pca.fit(data_train)
pca_train = pca.transform(data_train)
pca_test = pca.transform(data_test)

pca_model = fc_model_47()
pca_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
pca_model.fit(pca_train, labels_train, validation_data=(pca_test, labels_test), epochs=100, batch_size=200, shuffle=True)
pca_model.save('classifiers/fc-47-100-100-10-pca-model.h5')




