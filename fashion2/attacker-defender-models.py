#Written by: Rajeev Sahay

# The purpose of this code is to create the classifiers that both the
# attacker and defender will use in the experiments
# Defender classifier reaches accuracy of
# Attacker classifier reaches accuracy of

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import fashion_mnist

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



def fc_model_784_atkr():

    model = Sequential()
    model.add(Dense(200, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=784))
    model.add(Dense(200, activation="relu", kernel_initializer="normal"))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model

def fc_model_784():

    model = Sequential()
    model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=784))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model



#Build the attacker's model
model_784_atkr = fc_model_784_atkr()
#Compile model using cross entropy as loss and adam as optimizer
model_784_atkr.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
model_784_atkr.fit(data_train, labels_train, validation_data=(data_test, labels_test), epochs=100, batch_size=200, shuffle=True)
#Save the model
model_784_atkr.save('classifiers/fc-784-200-200-100-10-attacker-model.h5')


#Build the defender's model
model_784 = fc_model_784()
#Compile model using cross entropy as loss and adam as optimizer
model_784.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model
model_784.fit(data_train, labels_train, validation_data=(data_test, labels_test), epochs=100, batch_size=200, shuffle=True)
#Save the model
model_784.save('classifiers/fc-784-100-100-10-defender-model.h5')


def_acc = model_784.evaluate(data_test, labels_test)
print("Accuracy of defender classifier fc-100-100-10")
print ("Accuracy: %.2f%%" %(def_acc[1]*100))

atkr_acc = model_784_atkr.evaluate(data_test, labels_test)
print("Accuracy of attacker classifier fc-200-200-100-10")
print ("Accuracy: %.2f%%" %(atkr_acc[1]*100))
