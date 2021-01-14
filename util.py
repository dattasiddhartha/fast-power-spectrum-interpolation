import os
import scipy
import pyccl as ccl
import numpy as np
import pylab as plt
from numpy import linalg
import pandas as pd
import random

def feature_space(n_train = 10, n_k = 200, search_params = [5.0, 1.0, 1.0, 2.0, 2.0, 5.0, ]):
    """ 
    This function returns linearly-spaced value of dimension search. 
    """
    k_arr = np.logspace(-4., 2., n_k)
    oc_arr  = np.linspace(0.05, 1.0, int(search_params[0]*n_train))
    h_arr  = np.linspace(0.1,  1.0, int(search_params[1]*n_train))
    ob_arr = np.linspace(0.0, 0.3, int(search_params[2]*n_train))
    ns_arr = np.linspace(0.5, 1.5, int(search_params[3]*n_train))
    sigma8_arr = np.linspace(0.1, 2.0, int(search_params[4]*n_train))
    ts_arr = np.linspace(0.01, 1.0, int(search_params[5]*n_train))
    return k_arr, oc_arr, h_arr, ob_arr, ns_arr, sigma8_arr, ts_arr

def theoretical_Pk(k_arr, oc, h, ob, ns, sigma8, ts):
    """ Computes P(k) for input params"""
    cosmo = ccl.Cosmology(Omega_c=oc, Omega_b=ob, h=h,
                          sigma8=sigma8, n_s=ns)
    return ccl.power.linear_matter_power(cosmo, k_arr, ts)