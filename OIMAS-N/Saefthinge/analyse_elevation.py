import os
os.chdir('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/')
import sys
sys.path.append('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/')
import pandas as pd
from scipy.stats import linregress

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ip02')
import cmasher as cm
from matplotlib.patches import Patch


from rmse import rmse

from OIMAS import OIMAS_N

#%% load RTK elevation data
fn = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/RTK/sample_locations_RTK.csv"
elev = pd.read_csv(fn, skiprows=0, index_col=1)

elev["zone"] = [z.split('y')[0] for z in elev.index]

# customize boxplot properties
boxprops = dict(linestyle='-', linewidth=0, color='w')
medianprops = dict(linestyle='-', linewidth=1, color='w')
whiskerprops = dict(linestyle='-', linewidth=1, color='w')
capprops = dict(linewidth=0)

# initalize figure
fig, ax = plt.subplots()

elev_z = elev[["z_NAP","zone"]]
elev_by_zone = elev.groupby('zone')

# plot boxplot
elev_by_zone.boxplot(ax = ax, subplots = False, column = 'z_NAP',
                     boxprops = boxprops, medianprops = medianprops,
                     whiskerprops = whiskerprops, capprops = capprops,
                     rot = 0, vert = True, patch_artist = True)

ax.set_xticklabels(['\n'.join(t.get_text().split(',')[0][1:].split(' ')) for t in ax.get_xticklabels()])
ax.set_ylabel('RTK-derived elevation [m]')

#%% get statitistics

for _, group in elev_by_zone:
    print(group["z_NAP"].median())