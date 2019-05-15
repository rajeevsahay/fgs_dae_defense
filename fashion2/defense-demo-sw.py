#Written by: Rajeev Sahay 

# The purpose of this code is to test the various denoising autoencoder (DAE)
# arcitechtures as a defense against adversarial attacks in the semi-white box
# scenario for the MNIST fashion dataset

import numpy as np
import keras
from keras import backend
from keras.datasets import fashion_mnist
from keras.models import load_model
from keras import backend as K
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from sklearn.decomposition import PCA

#Perturbation magnitude. Adjust and run to observe behavior for various bounds.
eta = 0.25

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


#Import trained classifers
backend.set_learning_phase(False)
fc_classifier = load_model('classifiers/fc-784-100-100-10-defender-model.h5')
fc_16_100_100_10 = load_model('classifiers/fc-16-100-100-10-2550.h5')
fc_32_100_100_10 = load_model('classifiers/fc-32-100-100-10-2550.h5')
fc_47_100_100_10 = load_model('classifiers/fc-47-100-100-10-2550.h5')
fc_64_100_100_10 = load_model('classifiers/fc-64-100-100-10-2550.h5')
fc_80_100_100_10 = load_model('classifiers/fc-80-100-100-10-2550.h5')
fc_94_100_100_10 = load_model('classifiers/fc-94-100-100-10-2550.h5')
fc_157_100_100_10 = load_model('classifiers/fc-157-100-100-10-2550.h5')
adv_train_model = load_model('classifiers/fc-784-100-100-10-adv-trn.h5')
trad_ae_model = load_model('classifiers/fc-47-100-100-10-trad-ae.h5')
pca_model = load_model('classifiers/fc-47-100-100-10-pca-model.h5')

#Load DAEs
dae_16 = load_model('daes/dae_16_eps2550.h5')
dae_32 = load_model('daes/dae_32_eps2550.h5')
dae_47 = load_model('daes/dae_47_eps2550.h5')
dae_64 = load_model('daes/dae_64_eps2550.h5')
dae_80 = load_model('daes/dae_80_eps2550.h5')
dae_94 = load_model('daes/dae_94_eps2550.h5')
dae_157 = load_model('daes/dae_157_eps2550.h5')
trad_ae = load_model('daes/trad_ae.h5')

#Obtain reduced dimensional representation of adversarial data
get_hidden_layer_output_16 = K.function([dae_16.layers[0].input], [dae_16.layers[2].output])
get_hidden_layer_output_32 = K.function([dae_32.layers[0].input], [dae_32.layers[2].output])
get_hidden_layer_output_47 = K.function([dae_47.layers[0].input], [dae_47.layers[2].output])
get_hidden_layer_output_64 = K.function([dae_64.layers[0].input], [dae_64.layers[2].output])
get_hidden_layer_output_80 = K.function([dae_80.layers[0].input], [dae_80.layers[2].output])
get_hidden_layer_output_94 = K.function([dae_94.layers[0].input], [dae_94.layers[2].output])
get_hidden_layer_output_157 = K.function([dae_157.layers[0].input], [dae_157.layers[2].output])
trad_ae_output = K.function([trad_ae.layers[0].input], [trad_ae.layers[2].output])


#Create adversarial examples on testing data
sess =  backend.get_session()
wrap = KerasModelWrapper(fc_classifier)
fgsm = FastGradientMethod(wrap, sess=sess)
adv_test_x = fgsm.generate_np(data_test, eps=eta, clip_min=0., clip_max=1.)

wrap_adv_model = KerasModelWrapper(adv_train_model)
fgsm_adv_model = FastGradientMethod(wrap_adv_model, sess=sess)
adv_test_x_adv_model = fgsm_adv_model.generate_np(data_test, eps=eta, clip_min=0., clip_max=1.)


#Evaluate on clean data
scores = fc_classifier.evaluate(data_test, labels_test)
print("Accuracy of clean data without any defense")
print ("Accuracy: %.2f%%" %(scores[1]*100))

#Evaluate model after attacking data with no defense
adv_acc = fc_classifier.evaluate(adv_test_x, labels_test)
print("Accuracy of perturbed data without defense")
print ("Accuracy: %.2f%%" %(adv_acc[1]*100))

#Evaluate using traditional autoencoder hidden layer (k=47 dimensions)
trad_ae_hidden_47 = trad_ae_output([adv_test_x])[0]
trad_ae_acc = trad_ae_model.evaluate(trad_ae_hidden_47, labels_test)
print("Accuracy of perturbed data using traditional autoencoder dimensionality reduction (k=47)")
print ("Accuracy: %.2f%%" %(trad_ae_acc[1]*100))

#Evaluate accuracy of perturbed data using PCA dimensionality reduction (k=47)
#Calculate PCA projection
pca = PCA(47)
pca.fit(data_train)
pca_test = pca.transform(adv_test_x)
pca_acc = pca_model.evaluate(pca_test, labels_test)
print("Accuracy of perturbed data using PCA dimensionality reduction (k=47)")
print ("Accuracy: %.2f%%" %(pca_acc[1]*100))

#Evaluate accuracy of perturbed data uisng adversarially re-trained model
adv_model_acc = adv_train_model.evaluate(adv_test_x_adv_model, labels_test)
print("Accuracy of perturbed data using adversarially re-trained model")
print ("Accuracy: %.2f%%" %(adv_model_acc[1]*100))

#Evaluate accuracy of perturbed data after pre-processing
adv_test_x_16 = get_hidden_layer_output_16([adv_test_x])[0]
adv_scores_16 = fc_16_100_100_10.evaluate(adv_test_x_16, labels_test)
print("Accuracy of perturbed data using DAE hidden layer (k=16) as Defense")
print ("Accuracy: %.2f%%" %(adv_scores_16[1]*100))


adv_test_x_32 = get_hidden_layer_output_32([adv_test_x])[0]
adv_scores_32 = fc_32_100_100_10.evaluate(adv_test_x_32, labels_test)
print("Accuracy of perturbed data using DAE hidden layer (k=32) as Defense")
print ("Accuracy: %.2f%%" %(adv_scores_32[1]*100))


adv_test_x_47 = get_hidden_layer_output_47([adv_test_x])[0]
adv_scores_47 = fc_47_100_100_10.evaluate(adv_test_x_47, labels_test)
print("Accuracy of perturbed data using DAE hidden layer (k=47) as Defense")
print ("Accuracy: %.2f%%" %(adv_scores_47[1]*100))


adv_test_x_64 = get_hidden_layer_output_64([adv_test_x])[0]
adv_scores_64 = fc_64_100_100_10.evaluate(adv_test_x_64, labels_test)
print("Accuracy of perturbed data using DAE hidden layer (k=64) as Defense")
print ("Accuracy: %.2f%%" %(adv_scores_64[1]*100))


adv_test_x_80 = get_hidden_layer_output_80([adv_test_x])[0]
adv_scores_80 = fc_80_100_100_10.evaluate(adv_test_x_80, labels_test)
print("Accuracy of perturbed data using DAE hidden layer (k=80) as Defense")
print ("Accuracy: %.2f%%" %(adv_scores_80[1]*100))


adv_test_x_94 = get_hidden_layer_output_94([adv_test_x])[0]
adv_scores_94 = fc_94_100_100_10.evaluate(adv_test_x_94, labels_test)
print("Accuracy of perturbed data using DAE hidden layer (k=94) as Defense")
print ("Accuracy: %.2f%%" %(adv_scores_94[1]*100))


adv_test_x_157 = get_hidden_layer_output_157([adv_test_x])[0]
adv_scores_157 = fc_157_100_100_10.evaluate(adv_test_x_157, labels_test)
print("Accuracy of perturbed data using DAE hidden layer (k=157) as Defense")
print ("Accuracy: %.2f%%" %(adv_scores_157[1]*100))
