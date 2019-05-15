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
from keras.datasets import fashion_mnist
from keras.models import load_model
from keras import backend as K
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from sklearn.decomposition import PCA


e = np.linspace(0.01,0.50,50)
data = np.zeros([50, 8])

#Load training and testing data and normalize in [0, 1]
(data_train, labels_train), (data_test, labels_test) = fashion_mnist.load_data()
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
fc_classifier_47 = load_model('classifiers/fc-47-100-100-10-2550.h5')
pca_model = load_model('classifiers/fc-47-100-100-10-pca-model.h5')
adv_train_model = load_model('classifiers/fc-784-100-100-10-adv-trn.h5')
trad_ae_model = load_model('classifiers/fc-47-100-100-10-trad-ae.h5')
trad_ae = load_model('daes/trad_ae.h5')
pp_ae_eps2550_47 = load_model('daes/dae_47_eps2550.h5')

#Calculate PCA projection
pca = PCA(47)
pca.fit(data_train)


#Evaluate on clean data
scores = target_classifier.evaluate(data_test, labels_test)

#Setup adversairal attack for 784-100-100-10 clf
sess =  backend.get_session()
wrap = KerasModelWrapper(target_classifier)
fgsm = FastGradientMethod(wrap, sess=sess)

wrap_adv_trn = KerasModelWrapper(adv_train_model)
fgsm_adv_trn = FastGradientMethod(wrap_adv_trn, sess=sess)


#Obtain hidden layer representation of data from DAE
get_hidden_layer_output_47 = K.function([pp_ae_eps2550_47.layers[0].input], [pp_ae_eps2550_47.layers[2].output])

#Obtain hidden layer representation of data from traditional AE
trad_ae_output = K.function([trad_ae.layers[0].input], [trad_ae.layers[2].output])

#Retained accuracies
for idx, eta in enumerate(e):
    print(idx)
    data[idx, 0] = eta

    #Create adversarial examples on testing data
    adv_test_x = fgsm.generate_np(data_test, eps=eta, clip_min=0., clip_max=1.)

    #Evaluate accuracy without defense
    adv_acc = target_classifier.evaluate(adv_test_x, labels_test)
    data[idx, 1] = (adv_acc[1]/scores[1])*100


    #Get hidden layer representation of DAE of attacked data for k=47
    dae_hidden_adv_47 = get_hidden_layer_output_47([adv_test_x])[0]
    #Evaluate accuracy of adversarial data after extracting hidden representation
    dae_hid_layer_acc_47 = fc_classifier_47.evaluate(dae_hidden_adv_47, labels_test)
    data[idx,2] = (dae_hid_layer_acc_47[1]/scores[1])*100

    #Get hidden layer representation of traditional AE of attacked data
    trad_ae_hidden_47 = trad_ae_output([adv_test_x])[0]
    trad_ae_acc = trad_ae_model.evaluate(trad_ae_hidden_47, labels_test)
    data[idx, 3] = (trad_ae_acc[1]/scores[1])*100

    #PCA response
    pca_test = pca.transform(adv_test_x)
    pca_acc = pca_model.evaluate(pca_test, labels_test)
    data[idx, 4] = (pca_acc[1]/scores[1])*100

    adv_test_x_adv_trn = fgsm_adv_trn.generate_np(data_test, eps=eta, clip_min=0., clip_max=1.)
    #Adversarially trained model response
    adv_model_acc = adv_train_model.evaluate(adv_test_x_adv_trn, labels_test)
    data[idx, 5] = (adv_model_acc[1]/scores[1])*100

    #DAE output
    decoded_data = pp_ae_eps2550_47.predict(adv_test_x)
    dae_output_acc = target_classifier.evaluate(decoded_data, labels_test)
    data[idx, 6] = (dae_output_acc[1]/scores[1])*100

    #Cascade
    ae_hidden_layer = trad_ae_output([decoded_data])[0]
    cascade_acc = trad_ae_model.evaluate(ae_hidden_layer, labels_test)
    data[idx, 7] = (cascade_acc[1]/scores[1])*100





np.savetxt("retained_acc_wb_sigprocletter_digits.csv", data, delimiter=",")
