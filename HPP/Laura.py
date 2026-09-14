from scipy.interpolate import griddata
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from myselafin import Selafin
from datetime import datetime, timedelta
import matplotlib
matplotlib.style.use('ip02')
matplotlib.use('Agg')
#%% define functions
def get_index(x, y, tel):
    return griddata((tel.x, tel.y), np.arange(len(tel.x)), ([x],[y]), method='nearest')[0]

#%% import points of interest
gdf = gpd.read_file('/Users/ignace/Documents/Colleagues/Laura/Seedbank31300.shp', engine = 'pyogrio')
gdf.index = gdf['Name']
gdf['x'] = gdf.geometry.x
gdf['y'] = gdf.geometry.y

#%% load telemac results
slf = Selafin('/Users/ignace/Documents/WETCOAST/model/HPP/output/out_t2d.slf')
slf.import_header()
s = slf.import_data(vname = 'FREE SURFACE'.ljust(16))
u = slf.import_data(vname = 'VELOCITY U'.ljust(16))
v = slf.import_data(vname = 'VELOCITY V'.ljust(16))
U = np.sqrt(u**2 + v**2)
smax = np.max(s, axis = 1)
#%% plot
fig, ax = plt.subplots()
im = ax.tripcolor(slf.x, slf.y, slf.ikle-1, smax, vmax = 6)
ax.scatter(gdf.x, gdf.y, 10, c = 'C0')
for i in gdf.index:
    x = gdf.x.loc[i]
    y = gdf.y.loc[i]
    ax.annotate(str(i), (x, y), c = 'C0')
fig.colorbar(im, label = 'z [m]')

#%% extract data
T = [datetime(2025,5,1) + timedelta(seconds = t) for t in slf.times]

for i in gdf.index:
    x = gdf.x.loc[i]
    y = gdf.y.loc[i]
    ui = U[get_index(x, y, slf), :]
    si = s[get_index(x, y, slf), :]

    df = pd.DataFrame({'time': T, 'velocity_m/s': ui, 'water_level_mTAW': si})
    df.to_csv(f'/Users/ignace/Documents/Colleagues/Laura/{i}.csv', index=False)

    fig, ax = plt.subplots()
    ax.plot(T, ui, color = 'white', label = 'velocity')
    ax.set_ylabel('velocity [m/s]', zorder = 2)
    ax.set_ylim(-.1,.99)
    ax.set_title(f'{i}')
    # add twin ax
    ax2 = ax.twinx()
    ax2.plot(T, si, label = 'water level', color = 'C1', zorder = 1)
    ax2.set_ylabel('water level [m]', color = 'C1')
    ax2.grid(False)
    ax2.set_ylim(2.4, 5.9)
    # set C1 as color for ticks and ticklabels
    ax2.tick_params(axis='y', colors='C1')

    fig.tight_layout()
    fig.savefig(f'/Users/ignace/Documents/Colleagues/Laura/{i}.png')


