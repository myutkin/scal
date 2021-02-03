#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:22:27 2021

@author: yutkinm
"""

import pandas as pd
import numpy as np

"""
The import functions are unique to MetaRock file format

Example of a driver file

import numpy as np
from scipy import stats
from perm_functions import metarock_import, merge_pumps, uniq_flows, stable_dP, calc_perm

#%% Read in data, takes time if file is large
filename = 'YOURFILENAME'
data = metarock_import(filename)
data = merge_pumps(data)

#%% 
flow_rates = uniq_flows(data['Injc. Rate 2'],rounding=1,lim_len=10) # cc/min 
# alternatively can input flow manually

pressures = stable_dP(data['D.P. Paro Psi'], data['Injc. Rate 2'], flow_rates, rounding=1)

pressures = np.array(pressures) * 6894.76 # Pa
flow_rates = flow_rates * 1.66667e-8 # m3/s
mu = 1e-3 # Pa·s #fluid viscosity

L = np.array([4.45, 4.42, 4.40, 4.42, 4.43 ])/100 #core length, m
L_mean = np.mean(L)
D = np.array([3.804, 3.803, 3.802, 3.804, 3.805])/100 #core diameter, m
D_mean = np.mean(D)
A = 3.14*pow(D_mean,2)/4 #core end surface area, m2

perm = calc_perm(flow_rates, L_mean, pressures, mu, A)
#%% do lin fit

y = flow_rates / A
x = pressures / L / mu

res = stats.linregress(x,y)

print(r'Liquid permeability is {:.2f} ± {:.2f} mD'.format(res[0]*1000 / 1e-12, res[-1]*1000/1e-12))


"""

def metarock_import(filename, apply_offsets=True):
    
    header1 = pd.read_csv(filename, skiprows=29, sep=',', nrows=1, usecols = range(56), 
                      header=None,  encoding='ISO-8859-1',skipinitialspace=True)
    header1[0] = 'Date'
    header2 = pd.read_csv(filename, skiprows=30, sep=',', nrows=1, usecols = range(56), 
                      header=None, skipinitialspace=True,encoding='ISO-8859-1')
    header2[0] = 'days'

    HEADER = [str(i).strip() + ' ' + str(j).strip() for i,j in zip(header1.values.tolist()[0],header2.values.tolist()[0])]
    data = pd.read_csv(filename, skiprows=47, sep=',', header=None, usecols = range(56), names=HEADER)
    offsets = pd.read_csv(filename, skiprows=34, sep=',', nrows=1, usecols = range(56), 
                          header=None,  encoding='ISO-8859-1',skipinitialspace=True, names=HEADER)      
    
    if apply_offsets:
        for column in offsets.columns:
            data[column] = data[column].apply(lambda x: x + offsets[column])
        return data
    else:
        return data, offsets

def merge_pumps(data):
    data['Injc. 1 Enc. Ra CC/Min'] = data['Injc. 1 Enc. Ra CC/Min'].apply(lambda x: x if x >=0 else 0)
    data['Injc. 2 Enc. Ra CC/Min'] = data['Injc. 2 Enc. Ra CC/Min'].apply(lambda x: x if x >=0 else 0)
    data['Injc. 3 Enc. Ra CC/Min'] = data['Injc. 3 Enc. Ra CC/Min'].apply(lambda x: x if x >=0 else 0)
    data['Injc. 4 Enc. Ra CC/Min'] = data['Injc. 4 Enc. Ra CC/Min'].apply(lambda x: x if x >=0 else 0)
    data['Injc. Rate 1'] = data['Injc. 1 Enc. Ra CC/Min'] + data['Injc. 2 Enc. Ra CC/Min']
    data['Injc. Rate 2'] = data['Injc. 3 Enc. Ra CC/Min'] + data['Injc. 4 Enc. Ra CC/Min']
    return data

def uniq_flows(column, rounding=1, lim_len=10):
    uflows = np.unique(np.round(column, rounding))
    uflows = uflows[uflows > 0]
    uflows = [q for q in uflows if len(column[np.round(column,rounding) == q]) > lim_len]
    return np.array(uflows)

def sd_trim(data, sds):
    avg = np.mean(data)
    sd = np.std(data)
    norm = (data - avg)/ sd
    trim_idx = np.abs(norm) < sds
    data = data[trim_idx]
    return data

def stable_dP(dP, Q, u_rates, rounding=1, lim_len=10, sds=1):
    
    stable_pressures = list([np.array(dP[np.round(Q,rounding) == q]) for q in u_rates if len(dP[np.round(Q,rounding)==q]) > lim_len]) 
    stable_pressures  = [np.mean(sd_trim(region,sds)) for region in stable_pressures]
    return stable_pressures
    
def calc_perm(Q, L, dP, mu, A):
    # supply vars in SI
    perm = L * mu * Q/ (A * dP)
    return perm
        