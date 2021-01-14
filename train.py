import tensorflow as tf
import os
import time
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.utils import shuffle
from sklearn.metrics import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from model import *

def train(x_train, y_train,
          cycles = 40, batch_size = 1000, learning_rate = 0.001, 
          RNN__fname = "lstm6_tts10_bs1000", lstm_pretrained = True, lstm_pretrained_fname = 'lstm5_epch240_tts10_bs1000_129_17759.h5',
          k1 = 8e-7, k2 = 0.008, ae_epochs = 40,
          ae_e0_fname = "ae7_e0_tts10_bs100.h5", ae0_pretrained = False, ae0_pretrained_fname = '',
          ae_e1_fname = "ae7_e1_tts10_bs100.h5", ae1_pretrained = False, ae1_pretrained_fname = '',
          ae_e2_fname = "ae7_e2_tts10_bs100.h5", ae2_pretrained = False, ae2_pretrained_fname = '',
         ):
    """
    Training Attention-based LSTM with 3 ensembles of autoencoders
    """
    RNN = LSTM(lstm_pretrained, lstm_pretrained_fname)
    n_epochs = x_train.shape[0] // batch_size
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.optimizers.Adam(learning_rate)

    for cycle in range(cycles):
        X_train, Y_train = shuffle(x_train, y_train, random_state = cycle+111)
        for epoch in range(n_epochs):
            start = epoch * batch_size
            X_batch = X_train[start:start+batch_size, :]
            Y_batch = Y_train[start:start+batch_size, ]
            with tf.GradientTape() as tape:
                current_loss = loss(RNN(X_batch), Y_batch)
            gradients = tape.gradient(current_loss, RNN.trainable_variables)
            optimizer.apply_gradients(zip(gradients, RNN.trainable_variables))
            if (epoch+1) % 10 == 0:
                print("Epoch "
                      + str(epoch) 
                      + '.\tTraining Loss: '   
                      + str(current_loss.numpy()))
        weights_export(RNN, RNN__fname+str("_")+str(cycle)+str("_")+str(epoch)+'.h5')
    print('\nLSTM complete.')

    pre_P_k = RNN.predict(x_train)
    x_pre = pd.DataFrame(x_train.reshape(x_train.shape[0], x_train.shape[1]), columns=['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't'])
    x_pre['P_k_pre'] = pre_P_k
    x_pre['Ground_truth'] = y_train

    x_pre_e0 = x_pre[x_pre['k']<=k1] # cutoff between k
    x_pre_e2 = x_pre[x_pre['k']>k2] # cutoff between k
    pp = x_pre[x_pre['k']<=k2]
    x_pre_e1 = pp[pp['k']>k1] # cutoff between k

    #### e0
    output_seq = x_pre_e0[['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'Ground_truth']]
    input_seq = x_pre_e0[['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'P_k_pre']]

    input_seq_array = input_seq.astype(np.float32)
    input_seq_array = np.array(input_seq_array).reshape((input_seq_array.shape[0], input_seq_array.shape[1], 1))
    output_seq_array = output_seq.astype(np.float32)
    output_seq_array = np.array(output_seq_array).reshape((output_seq_array.shape[0], output_seq_array.shape[1], 1))

    ae0 = autoencoder(ae0_pretrained, ae0_pretrained_fname)
    ae0.compile(loss='mse', optimizer="adam")
    ae0.fit(input_seq_array, output_seq_array, epochs=ae_epochs, verbose=True)
    ae0.save_weights(ae_e0_fname)

    #### e1
    output_seq = x_pre_e1[['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'Ground_truth']]
    input_seq = x_pre_e1[['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'P_k_pre']]

    input_seq_array = input_seq.astype(np.float32)
    input_seq_array = np.array(input_seq_array).reshape((input_seq_array.shape[0], input_seq_array.shape[1], 1))
    output_seq_array = output_seq.astype(np.float32)
    output_seq_array = np.array(output_seq_array).reshape((output_seq_array.shape[0], output_seq_array.shape[1], 1))

    ae1 = autoencoder(ae1_pretrained, ae1_pretrained_fname)
    ae1.compile(loss='mse', optimizer="adam")
    ae1.fit(input_seq_array, output_seq_array, epochs=ae_epochs, verbose=True)
    ae1.save_weights(ae_e1_fname)

    #### e2
    output_seq = x_pre_e2[['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'Ground_truth']]
    input_seq = x_pre_e2[['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'P_k_pre']]

    input_seq_array = input_seq.astype(np.float32)
    input_seq_array = np.array(input_seq_array).reshape((input_seq_array.shape[0], input_seq_array.shape[1], 1))
    output_seq_array = output_seq.astype(np.float32)
    output_seq_array = np.array(output_seq_array).reshape((output_seq_array.shape[0], output_seq_array.shape[1], 1))
    
    ae2 = autoencoder(ae2_pretrained, ae2_pretrained_fname)
    ae2.compile(loss='mse', optimizer="adam")
    ae2.fit(input_seq_array, output_seq_array, epochs=ae_epochs, verbose=True)
    ae2.save_weights(ae_e2_fname)

    return RNN, ae0, ae1, ae2
