import tensorflow as tf
import os
import time
import math
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.utils import shuffle
from sklearn.metrics import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import sqlite3
iteration = 889100
conn = sqlite3.connect("cmb_export_iteration_"+str(iteration)+".db")
#############################################

def tolist(x):
    return np.array(literal_eval(x.replace(" ", ",")))

def tolog(x):
    return math.log(x)

def exploding_rows(cmb, chunksize=1000, iteration = 889100):
    """
    1000 splits rows by 1,000
    """
    exploded_dfs = [cmb[int((i-1)*chunksize):int(i*chunksize)].explode(cmb.columns[0]) for i in range(1, round(cmb.shape[0]/chunksize))]
    patch = [cmb[int((i-1)*chunksize):int(i*chunksize)].iloc[:, 1:].explode(cmb.iloc[:, 1:].columns[0])[cmb.iloc[:, 1:].columns[0]] for i in range(1, round(cmb.shape[0]/chunksize))]
    exploded_dfs_patched = []
    for i in range(len(patch)):
        exploded_dfs[i]['k'] = patch[i]
        exploded_dfs_patched.append(exploded_dfs[i])
        
    exploded_dfs_patched[0].to_sql("iteration_"+str(iteration), conn, if_exists="append")
    for i in range(1, len(exploded_dfs_patched)):
        exploded_dfs_patched[i].to_sql("iteration_"+str(iteration), conn, if_exists="append")
    faster = pd.read_sql_query("select * from "+str("iteration_"+str(iteration))+";", conn, chunksize=1000)
    concat_df = pd.concat(faster, ignore_index=True)
    return concat_df


def generation(iteration = 889100):
    """
    Returns raw, non-arrayed, exploded data
    """
    faster = pd.read_csv("cmb_export_"+str(iteration)+".csv", iterator=True, chunksize=1000, usecols = ['pk_true', 'k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't'])
    cmb = pd.concat(faster, ignore_index=True)
    # convert str to array
    cmb[cmb.columns[0]] = cmb[cmb.columns[0]].apply(tolist)
    cmb[cmb.columns[1]] = cmb[cmb.columns[1]].apply(tolist)
    print("cmb shape: ", cmb.shape)
    # explode P_k and k cols
    cmb_expl = exploding_rows(cmb, chunksize=1000, iteration=iteration)
    del cmb # free up memory with dataframe deletions
    print("Exploded shape: ", cmb_expl.shape)
    
    

def train_test_params(iteration = 889100, train_test_split_iter = 1, denomination = 1000):
    """
    Perform train-test split here, export train_params and test_params
    Perform test set evaluation on CPU server, while train set training on GPU server.
    """
    # sampling method
    faster = pd.read_sql_query("SELECT DISTINCT Omega_c, h, Omega_b, sigma8, n_s, t from "+str("iteration_"+str(iteration))+";", conn, chunksize=1000)
    cmb_unique = pd.concat(faster, ignore_index=True).reset_index(drop=True)
    optimal_train_sample_len = int(len(cmb_unique)*train_test_split_iter/denomination)
    train_unique = cmb_unique.sample(optimal_train_sample_len)
    test_unique = cmb_unique[~cmb_unique.index.isin(train_unique.index)]
    del cmb_unique
    return train_unique, test_unique
    
def data_load(set_unique, iteration = 889100):
    """
    Load sql database of pre-generated feature space (based on config in dataset_parallel.py)
    Performs post-processing for model (e.g. logarithmic P_k, min-max normalization)
    Export as either train or test
    
    Mode: 
    (1) train (requires train-test) ~ Usage: pass in set_unique dataframe as argument
    (2) eval (uses cmb directly, no need call train_test_params) ~ Usage: assign "eval" string to set_unique
    """
    faster = pd.read_sql_query("SELECT pk_true, k, Omega_c, h, Omega_b, sigma8, n_s, t FROM "+str("iteration_"+str(iteration))+";", conn, chunksize=1000)
    cmb = pd.concat(faster, ignore_index=True)
    # logarithmic transformation of P_k
    cmb[cmb.columns[0]] = cmb[cmb.columns[0]].apply(tolog)
    # set split/export
    if type(set_unique) == type(cmb):
        set_expl = cmb.merge(set_unique, how='inner', on=list(set_unique.columns))
        del cmb
    if type(set_unique) == type("eval"):
        set_expl = cmb
    # normalize the data -- check with train_set_expl.describe()
    max_norms = [max(set_expl[set_expl.columns[i]]) for i in range(len(set_expl.columns))]
    min_norms = [min(set_expl[set_expl.columns[i]]) for i in range(len(set_expl.columns))]
    i=0
    for col in set_expl.columns:
        set_expl[col] = (set_expl[col] - min_norms[i]) / (max_norms[i] - min_norms[i])
        i+=1
    return set_expl


def model_inputs(set_expl):
    """
    Converts training / testing data into tensor inputs to be processed by model.
    """
    x_set = set_expl.iloc[:, 1:]
    y_set = set_expl.iloc[:, 0]
    del set_expl
    x_set = x_set.astype(np.float32)
    y_set = y_set.astype(np.float32)
    # # reshape data as required by Keras LSTM
    x_set = np.array(x_set).reshape((x_set.shape[0], x_set.shape[1], 1)) # Number of observations ,  Window size  ,  Number of input series
    y_set = np.array(y_set).reshape((y_set.shape[0], 1))
    print("Tensor  x, y shape: ", x_set.shape, y_set.shape)
    return x_set, y_set
