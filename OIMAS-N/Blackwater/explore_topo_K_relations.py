import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.style.use("ip02")
from scipy.stats import linregress

BASE_MODEL_DIR = "/Users/ignace/Documents/WETCOAST/model/OIMAS-N"
sys.path.append(BASE_MODEL_DIR)
sys.path.append(BASE_MODEL_DIR + '/Blackwater/')

from load_data_Blackwater import *

#%% load callibrated parameters
call = pd.read_csv('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/Blackwater/callibration_output/callibrated_params.csv', index_col=0)
# load RTK data
rtk, sar = load_rtk_data()

call = call.join(rtk, how='left')
z = call['Elevation_m']

fig, axs = plt.subplots(nrows = 3, sharex = True, figsize = (5,8))

for i, var in zip(range(3), ['Kla', 'Kre', 'sed']):
    y = call[var]

    res = linregress(z, y)

    axs[i].scatter(z, y)
    #axs[i].set_xlim(0.30, 0.7)

    if res.pvalue < 0.05:
        axs[i].plot(np.sort(z), res.slope * np.sort(z) + res.intercept, color='C0', lw=1.5, ls='-')
    print(res.pvalue)

axs[0].set_ylabel('$K_{la}$')
axs[1].set_ylabel('$K_{re}$')
axs[2].set_ylabel('$k_{MARSED}$')

axs[-1].set_xlabel('elevation [m NADP]')

fig.tight_layout()

