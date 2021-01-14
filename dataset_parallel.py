import os
import scipy
import pyccl as ccl
import numpy as np
import pylab as plt
from numpy import linalg
import pandas as pd
import random
from util import *

mode = 'parallel_search'
iter_index = 99 # 0-99
print("Index :", iter_index)
export_dirfilename = "/mnt/zfsusers/sdatta/Desktop/cmb_expts/cmb_sdat/bin/cmb_export_parallel_withT_"+str(iter_index)+"_.csv"

k_arr, oc_arr, h_arr, ob_arr, ns_arr, sigma8_arr, ts_arr = feature_space(10, 200, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ]) # 100*10e6 data points; 400 nodes, 250,000 per node
count = 0; lent = int(len(oc_arr)*len(h_arr)*len(ob_arr)*len(ns_arr)*len(sigma8_arr)*len(ts_arr))

Output_list = [] # [pk_true, oc, h, b, sigma, n, t]

if mode == 'parallel_search':
    simulations = []
    for ts in ts_arr:
        for ns in ns_arr:
            for sigma8 in sigma8_arr:
                for ob in ob_arr:
                    for h in h_arr:
                        for oc in oc_arr:
                            simulations.append([k_arr, oc, h, ob, ns, sigma8, ts])                   
    simulations = simulations[int(iter_index*10000):int((iter_index+1)*10000)]

    for params in simulations:
        k_arr, oc, h, ob, ns, sigma8, ts = params
        try:
            pk_true = theoretical_Pk(k_arr, oc, h, ob, ns, sigma8, ts)
            Output_list.append([pk_true, k_arr, oc, h, ob, ns, sigma8, ts])
            print("Iteration: ", count, "/", lent, "; Params: ", oc, h, ob, ns, sigma8, ts)
            if (count % 10) == 0:
                viz = pd.DataFrame(Output_list)
                viz.columns = ['pk_true', 'k', 'Omega_c', 'h', 'Omega_b', 'sigma8', 'n_s', 't']
                viz.to_csv(export_dirfilename)
        except:
            print("Failed config: ", oc, h, ob, ns, sigma8, ts)
            continue
        count+=1
        
if mode == "combine_dataframe":
    fnms = ["./data/"+lir for lir in os.listdir("./data/")]
    pd_df = 1; df = pd.read_csv(fnms[0])
    while pd_df < len(fnms):
        df = pd.concat([df, pd.read_csv(fnms[pd_df])], axis=0)
        print(pd_df, df.shape)
        pd_df+=1
    print("cmb_export_"+str(df.shape[0])+".csv")
    df.to_csv("cmb_export_"+str(df.shape[0])+".csv")
