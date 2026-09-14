import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.style.use("ip01")
import rasterio
from rasterstats import zonal_stats
from rasterio.plot import show
from scipy.stats import linregress
from itertools import combinations
from scipy import stats
#%% Load data

# load callibrated parameters
call = pd.read_csv('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/Saefthinge/callibration_output/callibrated_params.csv', index_col=0)

#%% load biomass data
biomass = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/biomass/summary.csv', index_col=0)

# load RTK data
rtk = gpd.read_file('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/RTK/sample_locations_topo_indices.gpkg', engine = 'pyogrio')
rtk.index = rtk['Point name']

# load LiDAR DTM
dtm_fn = '/Users/ignace/Documents/WETCOAST/Data/Saefthinge/LiDAR/DTM_40cm_Saeftinghe_Zomer25.tif'
dtm = rasterio.open(dtm_fn)
dtm_arr = dtm.read(1)

# join two dataframes
df = call.join(rtk[['x','y','z_TAW','dist_to_channel_manual', 'dist_to_5m', 'ruggedness', '3m_buffer_mean',
       '3m_buffer_median', '3m_buffer_min', '3m_buffer_max', '3m_buffer_range']], lsuffix= '', rsuffix= '', how = 'inner')

df = df.join(biomass[['primary_plant']], lsuffix= '', rsuffix= '', how = 'inner')

# add age column
df['age'] = [int(i.split('y')[0][1:]) for i in df.index]

# turn into geodataframe
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs='epsg:31370')

# load drone-borne LiDAR DTM
dtm_fn = '/Users/ignace/Documents/WETCOAST/Data/Saefthinge/LiDAR/DTM_40cm_Saeftinghe_Zomer25.tif'
dtm = rasterio.open(dtm_fn)
dtm_arr = dtm.read(1)

#%% add topographical indices

# sample dtm with gdf
gdf["z_LiDAR"] = [z[0]/100 for z in dtm.sample([(x,y) for x,y in zip(gdf.x, gdf.y)])]

gdf_buf = gpd.GeoDataFrame(geometry = gdf.buffer(2))

gdf = gdf.join(pd.DataFrame(zonal_stats(vectors=gdf_buf['geometry'], raster=dtm_fn, stats=['max','min','range'], suffix = 'buf_', nodata = -999)), how='left')
gdf['plant'] = pd.factorize(gdf['primary_plant'])[0]
# get subdataframes
S10 = gdf.loc[df.age == 10]
S20 = gdf.loc[df.age == 20]
S40 = gdf.loc[df.age == 40]


#%% plot against topographical indices
fig, axs = plt.subplots(ncols=4, nrows=3, sharex=True, figsize = (8, 5))

for col, (data, color) in enumerate(zip([S10, S20, S40, gdf], ['C0', 'orange', 'C2', 'k'])):
    z = data['z_TAW']

    for row, y in enumerate([data['Kla'], data['Kre'], np.log(data['sed'])]):
        res = linregress(z,y)
        axs[row, col].scatter(z, y, 10, color)
        #axs[row, col].scatter(z, y, 10, data['primary_plant'].factorize()[0], cmap = 'Set1')
        axs[row, col].text(0.975, 0.3, f"$R^2 = {res.rvalue**2:.2f}$\np = {res.pvalue:.2f}", ha='right', va='top', transform=axs[row, col].transAxes, fontsize=9, c = color)
        if res.pvalue < 0.05:
            axs[row, col].plot(np.sort(z), res.slope * np.sort(z) + res.intercept, color=color, lw=1.5, ls='-',
                               alpha=.5)
        #for i in range(len(z)):
        #    axs[row, col].text(z[i], y[i], data.index[i], ha='center', va='center', fontsize=8, c = color, alpha = .5)

# update axes with labels and ticks
#for a in axs[:,:].flat: a.set_xlim(4.7,5.45)
for a in axs[0, :]: a.set_ylim(-0.05, 0.25)
for a in axs[1, :]: a.set_ylim(-0.005, 0.06)
for a in axs[2, :]: a.set_ylim(-3, 1.)
for a in axs[:, 1:].flat: a.set_yticklabels([])
for a in axs[:, :].flat: a.set_xlim(4.55,5.65)
axs[0, 0].set_ylabel(r'$K_{la}$ [$year^{-1}$]')
axs[1, 0].set_ylabel(r'$K_{re}$ [$year^{-1}$]')
axs[2, 0].set_ylabel(r'$log(k)$')
for a in axs[2, :]: a.set_xlabel(r'$z$ [$m\;TAW$]')

# set titles
axs[0, 0].set_title('10 year zone')
axs[0, 1].set_title('20 year zone')
axs[0, 2].set_title('40 year zone')

fig.tight_layout()
fig.show()
# fig.savefig('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/Saefthinge/callibration_output/callibration_sed.png', dpi = 300)

