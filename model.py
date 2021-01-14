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

class Attention(Layer):    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()
    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        super(Attention,self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)

def LSTM(lstm_pretrained = True, lstm_pretrained_fname = 'lstm5_epch240_tts10_bs1000_129_17759.h5'):
    """
    If using a pretrained model, 
    set lstm_pretrained = True, 
    and set lstm_pretrained_fname to the pretrained weights.
    """
    RNN = Sequential([ 
            LSTM(90, return_sequences=True, 
                 input_shape = (7, 1)), 
            Attention(return_sequences=True),
            LSTM(90), 
            Dense(1, activation = None) 
        ])
    if lstm_pretrained == True:
        RNN.load_weights(pretrained_fname)
    return RNN

def autoencoder(ae_pretrained = False, ae_pretrained_fname = ''):
    ae = tf.keras.Sequential()
    ae.add(tf.keras.layers.Dense(units = input_seq_array.shape[1], activation = 'linear', input_shape=[input_seq_array.shape[1],input_seq_array.shape[2]]))
    ae.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
    ae.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
    ae.add(tf.keras.layers.Dense(units = 32, activation = 'relu'))
    ae.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
    ae.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
    ae.add(tf.keras.layers.Dense(units = 1, activation = 'linear'))
    if ae_pretrained == True:
        ae.load_weights(ae_pretrained_fname)
    return ae

