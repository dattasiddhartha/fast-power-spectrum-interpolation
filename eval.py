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

def prediction(x_eval, y_eval,
               lstm_pretrained = True, lstm_pretrained_fname = 'lstm5_epch240_tts10_bs1000_129_17759.h5',
               ae0_pretrained = True, ae1_pretrained = True, ae2_pretrained = True, 
               ae_e0_fname = 'ae6_e0_tts10_bs100.h5', ae_e1_fname = 'ae6_e1_tts10_bs100.h5', ae_e2_fname = 'ae3_e2_tts10_bs100.h5',
               k1 = 8e-7, k2 = 0.008, 
              ):
    
    RNN_ = LSTM(lstm_pretrained, lstm_pretrained_fname)
    ae0 = autoencoder(ae0_pretrained, ae_e0_fname)
    ae1 = autoencoder(ae1_pretrained, ae_e1_fname)
    ae2 = autoencoder(ae2_pretrained, ae_e2_fname)
    
    pre_P_k = RNN_.predict(x_eval)
    x_pre = pd.DataFrame(x_eval.reshape(x_eval.shape[0], x_eval.shape[1]), columns=['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't'])
    x_pre['P_k_pre'] = pre_P_k
    x_pre['gtruth'] = y_eval
    
    x_pre_e0 = x_pre[x_pre['k']<=k1] # cutoff between k
    x_pre_e2 = x_pre[x_pre['k']>k2] # cutoff between k
    pp = x_pre[x_pre['k']<=k2]
    x_pre_e1 = pp[pp['k']>k1] # cutoff between k
    
    input_seq = x_pre_e0[['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'P_k_pre']]
    input_seq_array = input_seq.astype(np.float32)
    input_seq_array = np.array(input_seq_array).reshape((input_seq_array.shape[0], input_seq_array.shape[1], 1))
    P_k_ae = ae1.predict(input_seq_array)
    P_k_ae_e1 = pd.DataFrame(P_k_ae.reshape(P_k_ae.shape[0], P_k_ae.shape[1]), columns=['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'P_k_ae'])
    
    input_seq = x_pre_e1[['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'P_k_pre']]
    input_seq_array = input_seq.astype(np.float32)
    input_seq_array = np.array(input_seq_array).reshape((input_seq_array.shape[0], input_seq_array.shape[1], 1))
    P_k_ae = ae2.predict(input_seq_array)
    P_k_ae_e2 = pd.DataFrame(P_k_ae.reshape(P_k_ae.shape[0], P_k_ae.shape[1]), columns=['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'P_k_ae'])
    
    input_seq = x_pre_e2[['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'P_k_pre']]
    input_seq_array = input_seq.astype(np.float32)
    input_seq_array = np.array(input_seq_array).reshape((input_seq_array.shape[0], input_seq_array.shape[1], 1))
    P_k_ae = ae3.predict(input_seq_array)
    P_k_ae_e3 = pd.DataFrame(P_k_ae.reshape(P_k_ae.shape[0], P_k_ae.shape[1]), columns=['k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't', 'P_k_ae'])
    
    x_pre_e0_df['P_k'] = list(P_k_ae0['P_k_ae'])
    x_pre_e1_df['P_k'] = list(P_k_ae1['P_k_ae'])
    x_pre_e2_df['P_k'] = list(P_k_ae2['P_k_ae'])

    x_eN = pd.concat([x_pre_e0_df, x_pre_e1_df], ignore_index=True).sort_values(by='k')
    x_eN = pd.concat([x_eN, x_pre_e2_df], ignore_index=True).sort_values(by='k')
    
    return x_eN, x_pre_e0, x_pre_e1, x_pre_e2, P_k_ae_e1, P_k_ae_e2, P_k_ae_e3
    
def accuracy_test():
    
length = round(len(x_eval)/10000)
accList1 = []; accList2 = []; accList3 = [];
for i in range(1, length):
    start = int((i-1)*10000)
    end = int(i*10000)
    x_sample = x_eval[start:end]
    y_sample = y_eval[start:end]
    x_eN, x_pre_e0, x_pre_e1, x_pre_e2, P_k_ae_e1, P_k_ae_e2, P_k_ae_e3 = prediction(x_eval, y_eval,
               lstm_pretrained = True, lstm_pretrained_fname = 'lstm5_epch240_tts10_bs1000_129_17759.h5',
               ae0_pretrained = True, ae1_pretrained = True, ae2_pretrained = True, 
               ae_e0_fname = 'ae6_e0_tts10_bs100.h5', ae_e1_fname = 'ae6_e1_tts10_bs100.h5', ae_e2_fname = 'ae3_e2_tts10_bs100.h5',
               k1 = 8e-7, k2 = 0.008)
    print(start, end)
    acc1 = r2_score(x_pre_e0['gtruth'], P_k_ae_e1['P_k_ae']); accList1.append(acc1)
    acc2 = r2_score(x_pre_e1['gtruth'], P_k_ae_e2['P_k_ae']); accList2.append(acc2)
    acc3 = r2_score(x_pre_e2['gtruth'], P_k_ae_e3['P_k_ae']); accList3.append(acc3)
    print("R2(e1, e2): ", acc1, acc2, acc3)
    
    return accList1, accList2, accList3


