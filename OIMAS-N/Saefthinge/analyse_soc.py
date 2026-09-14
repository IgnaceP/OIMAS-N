import os
import sys

BASE_MODEL_DIR = "/Users/ignace/Documents/WETCOAST/model/OIMAS-N/"
os.chdir(BASE_MODEL_DIR)
sys.path.append(BASE_MODEL_DIR)
sys.path.append(BASE_MODEL_DIR + 'Saefthinge/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ip02')
matplotlib.use('MacOSX')

from functions import load_soil_carbon
from read_C_obs_data import read_observation_data

#%% Load data

soil    = load_soil_carbon([0, 40], read_observation_data)
S40y    = soil[40]

rtk     = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/RTK/sample_locations_RTK.csv', index_col=1)[["z_TAW"]]
sar     = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/LiDAR/surface_elevation_accumulation.csv',index_col=0)[['SAR']]
S40y    = S40y.join(rtk, how='left').join(sar, how='left')

S40y['hor40y']      = 40*S40y['SAR']
S40y['maxdepth']    = S40y.groupby(S40y.index)['depth'].transform(max)
S40y['C_perdepth']  = 0.05 * 1000* S40y['DBD'] * S40y['C_percentage']/100
S40y                = S40y.loc[S40y['maxdepth']/100 > S40y['hor40y']]
S40y['C_int']       = S40y.groupby(S40y.index)['C_perdepth'].transform(sum)
S40y['OCAR']        = S40y['C_int'] / 40

ocar                = S40y.groupby(S40y.index)[['OCAR','z_TAW']].mean()
S40y                = S40y.loc[S40y['depth']/100 < 40*S40y['SAR']]
