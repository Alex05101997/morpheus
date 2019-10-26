# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:49:49 2019

@author: Alex
"""

from python_speech_features import mfcc, delta
from scipy.io.wavfile import read
import numpy as np
import pandas as pd
import os
mfcc_feat = []
mfcc_delta = []
mfcc_delta_delta = []
rows_mfcc = []
rows_delta = []
rows_delta_delta = []
for filename in os.listdir('.'):
    if filename.endswith('.wav') and filename.startswith('G'):
        dict_mfcc = {}
        dict_delta = {}
        dict_delta_delta = {}
        fs,wav = read(filename)
        mfcc_coeff = mfcc(wav, fs, lowfreq=80, appendEnergy=False, ceplifter=0,winfunc=np.hamming)
        delta_coeff = delta(mfcc_coeff,1)
        delta_delta_coeff = delta(delta_coeff,1)
        mfcc_mean = np.mean(mfcc_coeff,axis=0)
        delta_coeff_mean = np.mean(delta_coeff,axis=0)
        delta_delta_coeff_mean = np.mean(delta_delta_coeff,axis=0)
        mfcc_z = (mfcc_mean-np.mean(mfcc_mean))/np.std(mfcc_mean)
        delta_z = (delta_coeff_mean-np.mean(delta_coeff_mean))/np.std(delta_coeff_mean)
        delta_delta_z = (delta_delta_coeff_mean-np.mean(delta_delta_coeff_mean))/np.std(delta_delta_coeff_mean)
        mfcc_feat = list(mfcc_z)
        mfcc_delta = mfcc_feat + list(delta_z)
        mfcc_delta_delta = mfcc_delta + list(delta_delta_z)
        dict_mfcc = {i : mfcc_feat[i] for i in range(0, len(mfcc_feat))}
        dict_delta = {i : mfcc_delta[i] for i in range(0, len(mfcc_delta))}    
        dict_delta_delta = {i : mfcc_delta_delta[i] for i in range(0, len(mfcc_delta_delta))}
        if filename.startswith('GA'):
            dict_mfcc.update({'Class':0})
            dict_delta.update({'Class':0})
            dict_delta_delta.update({'Class':0})
        else:
            dict_mfcc.update({'Class':1})
            dict_delta.update({'Class':1})
            dict_delta_delta.update({'Class':1})
            
        rows_mfcc.append(dict_mfcc)
        rows_delta.append(dict_delta)
        rows_delta_delta.append(dict_delta_delta)
        
        


df_1 = pd.DataFrame(rows_mfcc)
df_2 = pd.DataFrame(rows_delta)
df_3 = pd.DataFrame(rows_delta_delta)
#print(df)
writer = pd.ExcelWriter('features.xlsx', engine='xlsxwriter')
df_1.to_excel(writer, sheet_name='MFCC',index=False)
df_2.to_excel(writer, sheet_name='MFCC-DELTA',index=False)
df_3.to_excel(writer, sheet_name='MFCC-DELTA-DELTA',index=False)
writer.save()
