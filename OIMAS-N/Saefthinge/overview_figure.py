import os
import sys
BASE_MODEL_DIR = "/Users/ignace/Documents/WETCOAST/model/OIMAS-N"
os.chdir(BASE_MODEL_DIR)
sys.path.append(BASE_MODEL_DIR)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.style.use("ip01")
import rasterio
from rasterio.plot import show
from functions import run_model

zone = 'S40y'
call_df = pd.read_csv('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/callibrated_params.csv', index_col=0)
df = call_df.loc[[i for i in call_df.index if zone in i], :]
rtk = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/RTK/sample_locations_RTK.csv', index_col=1)
df = df.join(rtk[['z_TAW']], lsuffix= '', rsuffix= '', how = 'inner')
df = df.sort_values('z_TAW')

fig, axs = plt.subplots(nrows = 2, ncols = 4, figsize = (18,12), sharex=True, sharey=True)
axs = axs.flatten()
for i, auger_ID in enumerate(df.index):

    run_model(auger_ID, df.loc[auger_ID, 'Kla'],
              df.loc[auger_ID, 'Kre'], df.loc[auger_ID, 'sed'],
              ax = axs[i], c = 'C2')

fig.tight_layout()
fig.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/overview_{zone}.png')
plt.show()
