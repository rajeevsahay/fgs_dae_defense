#no defense (2nd column)
#DAE hidden layer (3rd column)
#trad AE hidden layer (4th column)
#PCA, first perturb data according to fc-200-200-100-10, then tranform using PCA(inside data collection loop) (5th column)
#adv training (6th column)
#DAE output
#Cascade

import numpy as np
import keras
from keras import backend
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from sklearn.decomposition import PCA


e = np.linspace(0.01,0.50,50)
data = np.zeros([50, 9])

#Load training and testing data and normalize in [0, 1]
(data_train, labels_train), (data_test, labels_test) = mnist.load_data()
data_train = data_train/255.0
data_test = data_test/255.0

#Flatten dataset (New shape for training and testing set is (60000,784) and (10000, 784))
data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))

#Create labels as one-hot vectors
labels_train = keras.utils.np_utils.to_categorical(labels_train, num_classes=10)
labels_test = keras.utils.np_utils.to_categorical(labels_test, num_classes=10)


#Import trained classifer
backend.set_learning_phase(False)


#Load models
target_classifier = load_model('classifiers/fc-784-100-100-10-defender-model.h5')
fc_classifier_16 = load_model('classifiers/fc-16-100-100-10-2550.h5')
fc_classifier_32 = load_model('classifiers/fc-32-100-100-10-2550.h5')
fc_classifier_47 = load_model('classifiers/fc-47-100-100-10-2550.h5')
fc_classifier_64 = load_model('classifiers/fc-64-100-100-10-2550.h5')
fc_classifier_80 = load_model('classifiers/fc-80-100-100-10-2550.h5')
fc_classifier_94 = load_model('classifiers/fc-94-100-100-10-2550.h5')
fc_classifier_157 = load_model('classifiers/fc-157-100-100-10-2550.h5')

pp_ae_eps2550_16 = load_model('daes/dae_16_eps2550.h5')
pp_ae_eps2550_32 = load_model('daes/dae_32_eps2550.h5')
pp_ae_eps2550_47 = load_model('daes/dae_47_eps2550.h5')
pp_ae_eps2550_64 = load_model('daes/dae_64_eps2550.h5')
pp_ae_eps2550_80 = load_model('daes/dae_80_eps2550.h5')
pp_ae_eps2550_94 = load_model('daes/dae_94_eps2550.h5')
pp_ae_eps2550_157 = load_model('daes/dae_157_eps2550.h5')


#Evaluate on clean data
scores = target_classifier.evaluate(data_test, labels_test)

#Setup adversairal attack for 784-100-100-10 clf
sess =  backend.get_session()
wrap = KerasModelWrapper(target_classifier)
fgsm = FastGradientMethod(wrap, sess=sess)


#Obtain hidden layer representation of data from DAE
get_hidden_layer_output_16 = K.function([pp_ae_eps2550_16.layers[0].input], [pp_ae_eps2550_16.layers[2].output])
get_hidden_layer_output_32 = K.function([pp_ae_eps2550_32.layers[0].input], [pp_ae_eps2550_32.layers[2].output])
get_hidden_layer_output_47 = K.function([pp_ae_eps2550_47.layers[0].input], [pp_ae_eps2550_47.layers[2].output])
get_hidden_layer_output_64 = K.function([pp_ae_eps2550_64.layers[0].input], [pp_ae_eps2550_64.layers[2].output])
get_hidden_layer_output_80 = K.function([pp_ae_eps2550_80.layers[0].input], [pp_ae_eps2550_80.layers[2].output])
get_hidden_layer_output_94 = K.function([pp_ae_eps2550_94.layers[0].input], [pp_ae_eps2550_94.layers[2].output])
get_hidden_layer_output_157 = K.function([pp_ae_eps2550_157.layers[0].input], [pp_ae_eps2550_157.layers[2].output])



#Retained accuracies
for idx, eta in enumerate(e):
    print(idx)
    data[idx, 0] = eta

    #Create adversarial examples on testing data
    adv_test_x = fgsm.generate_np(data_test, eps=eta, clip_min=0., clip_max=1.)

    no_def = target_classifier.evaluate(adv_test_x, labels_test)
    data[idx, 1] = (no_def[1]/scores[1])*100

    #Get hidden layer representation of DAE of attacked data for k=16
    dae_hidden_adv_16 = get_hidden_layer_output_16([adv_test_x])[0]
    #Evaluate accuracy of adversarial data after extracting hidden representation
    dae_hid_layer_acc_16 = fc_classifier_16.evaluate(dae_hidden_adv_16, labels_test)
    data[idx,2] = (dae_hid_layer_acc_16[1]/scores[1])*100

    #Get hidden layer representation of DAE of attacked data for k=32
    dae_hidden_adv_32 = get_hidden_layer_output_32([adv_test_x])[0]
    #Evaluate accuracy of adversarial data after extracting hidden representation
    dae_hid_layer_acc_32 = fc_classifier_32.evaluate(dae_hidden_adv_32, labels_test)
    data[idx,3] = (dae_hid_layer_acc_32[1]/scores[1])*100

    #Get hidden layer representation of DAE of attacked data for k=47
    dae_hidden_adv_47 = get_hidden_layer_output_47([adv_test_x])[0]
    #Evaluate accuracy of adversarial data after extracting hidden representation
    dae_hid_layer_acc_47 = fc_classifier_47.evaluate(dae_hidden_adv_47, labels_test)
    data[idx,4] = (dae_hid_layer_acc_47[1]/scores[1])*100

    #Get hidden layer representation of DAE of attacked data for k=64
    dae_hidden_adv_64 = get_hidden_layer_output_64([adv_test_x])[0]
    #Evaluate accuracy of adversarial data after extracting hidden representation
    dae_hid_layer_acc_64 = fc_classifier_64.evaluate(dae_hidden_adv_64, labels_test)
    data[idx,5] = (dae_hid_layer_acc_64[1]/scores[1])*100

    #Get hidden layer representation of DAE of attacked data for k=80
    dae_hidden_adv_80 = get_hidden_layer_output_80([adv_test_x])[0]
    #Evaluate accuracy of adversarial data after extracting hidden representation
    dae_hid_layer_acc_80 = fc_classifier_80.evaluate(dae_hidden_adv_80, labels_test)
    data[idx,6] = (dae_hid_layer_acc_80[1]/scores[1])*100

    #Get hidden layer representation of DAE of attacked data for k=94
    dae_hidden_adv_94 = get_hidden_layer_output_94([adv_test_x])[0]
    #Evaluate accuracy of adversarial data after extracting hidden representation
    dae_hid_layer_acc_94 = fc_classifier_94.evaluate(dae_hidden_adv_94, labels_test)
    data[idx,7] = (dae_hid_layer_acc_94[1]/scores[1])*100

    #Get hidden layer representation of DAE of attacked data for k=157
    dae_hidden_adv_157 = get_hidden_layer_output_157([adv_test_x])[0]
    #Evaluate accuracy of adversarial data after extracting hidden representation
    dae_hid_layer_acc_157 = fc_classifier_157.evaluate(dae_hidden_adv_157, labels_test)
    data[idx,8] = (dae_hid_layer_acc_157[1]/scores[1])*100




np.savetxt("average_accuracies_wb.csv", data, delimiter=",")

averages = np.mean(data, axis=0, dtype=np.float64)
np.savetxt("averages.csv", averages, delimiter=",")

print(averages )
