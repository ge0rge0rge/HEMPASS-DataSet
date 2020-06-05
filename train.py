# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 00:32:20 2019

@author: wzqfox
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

train = pd.read_csv('all_train.csv.gz')
dataTest = pd.read_csv('all_test.csv.gz')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train['mass'] = scaler.fit_transform(train['mass'].values.reshape(-1,1))
dataTest['mass'] = scaler.fit_transform(dataTest['mass'].values.reshape(-1,1))

from sklearn.model_selection import train_test_split
X = train.drop(['# label'], axis=1)
y = train['# label']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.info()
Z = dataTest.drop(['# label'], axis=1)
completeDataTest = np.array(dataTest,dtype ='float32')
lablesTest = completeDataTest[:,0:1]
lablesTest=np.reshape(lablesTest,3500000)
completeDataTest = completeDataTest[:,1:29]
num_labels = 1
lablesTest = (np.arange(num_labels) == lablesTest[:,None]).astype(np.float32)
test_labels = lablesTest[:3500000]
Z.shape

dataTest.info()

from keras.regularizers import l2 # L2 regularization
from keras.callbacks import EarlyStopping
from keras.optimizers import *
from keras.callbacks import ReduceLROnPlateau
def create_deep_neural_net(num_inputs, hidden_layer_sizes, l2_val, num_outputs, optimizer):
    
    model = Sequential()
    first = True
    for hidden_layer_size in hidden_layer_sizes:
        if first:
            model.add(Dense(hidden_layer_size, 
                        activation='sigmoid', 
                        input_dim=num_inputs, kernel_regularizer=l2(l2_val)))
            first = False
        else:
            model.add(Dense(hidden_layer_size, 
                        activation='sigmoid', 
                        kernel_regularizer=l2(l2_val)))
        
    model.add(Dense(num_outputs, activation='sigmoid'))


    # compiling model
    model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    return model


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=1, min_lr=0, verbose=1, workers=8)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='min')

L2_CONSTANT = 1e-7 # L2 regularization constant
layers = [28] * 5 # 5 layers of size 28

sgd = SGD(lr=0.01, momentum=0.95)

callbacks = [reduce_lr, early_stopping]


uniform_width_network = create_deep_neural_net(num_inputs=28, 
                                      hidden_layer_sizes=layers,
                                      l2_val = L2_CONSTANT,
                                      num_outputs=1,
                                      optimizer=sgd)

print(uniform_width_network.summary())
uniform_width_history = uniform_width_network.fit(X_train.values, y_train.values,
validation_data=(X_valid.values, y_valid.values), callbacks=callbacks, epochs=50, batch_size=512)

test_acc,test_loss = uniform_width_network.evaluate(Z,test_labels)
print('Test accuracy:', test_acc)

uniform_width_history.history['val_accuracy']





# =============================================================================
# import matplotlib.pyplot as plt
# early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, mode='min')
# val_accs = []
# for num_layers in range(1, 8):
#     L2_CONSTANT = 1e-7 # L2 regularization constant
#     layers = [28] * num_layers # 5 layers of size 28
# 
#     uniform_width_network = create_deep_neural_net(num_inputs=28, 
#                                       hidden_layer_sizes=layers,
#                                       l2_val = L2_CONSTANT,
#                                       num_outputs=1,
#                                       optimizer=sgd)
#     if num_layers == 1:
#         print('Fitting neural network with 1 hidden layer')
#     else:
#         print('Fitting neural network with {} hidden layers'.format(num_layers))
#     print(uniform_width_network.summary())
#     uniform_width_history = uniform_width_network.fit(X_train.values, y_train.values, 
#                  validation_data=(X_valid.values, y_valid.values), epochs=25, 
#                                                       batch_size=256,
#                                                      callbacks=[early_stopping])
#     val_accs.append(uniform_width_history.history['val_accuracy'][-1])
# 
# plt.title('Validation accuracy with different layer depths')
# plt.xlabel('Number of hidden layers')
# plt.ylabel('Validation accuracy')
# plt.plot(range(1, 8), val_accs)
# =============================================================================


# =============================================================================
# def create_witm(num_inputs, l2_val, width_factor, depth_factor, num_outputs, optimizer):
#     
#     layers = []
#     for i in range(depth_factor+1):
#         layers.append((width_factor ** i) * num_inputs)
#     
#     for i in range(depth_factor):
#         layers.append(layers[-1] // width_factor)
#     
#     print(layers)
#     witm_network = create_deep_neural_net(num_inputs=num_inputs,
#                                          hidden_layer_sizes=layers,
#                                          l2_val=l2_val,
#                                          num_outputs=num_outputs,
#                                          optimizer=optimizer)
#     return witm_network
# 
# 
# L2_CONSTANT = 1e-7 # L2 regularization constant
# val_accs_witm = []
# 
# depth_factor=1
# 
# print('Fitting network with depth factor = {}...'.format(depth_factor))
# witm_network = create_witm(num_inputs=28,
#                               l2_val=L2_CONSTANT,
#                               width_factor=2,
#                               depth_factor=depth_factor,
#                               num_outputs=1,
#                               optimizer=sgd)
# print(witm_network.summary())
#     # fitting model
# history = witm_network.fit(X_train.values, y_train.values, validation_data=(X_valid.values, y_valid.values), epochs=100, 
#                  batch_size=512, callbacks=callbacks)
# val_accs_witm.append(max(history.history['val_accuracy']))
# history.history['val_accuracy']
# 
# test_loss1, test_acc1 = witm_network.evaluate(Z, test_labels)
# print('Test accuracy:', test_acc1)
# =============================================================================