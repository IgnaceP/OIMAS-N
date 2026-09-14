import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from pyproj import Transformer

import datetime

#%% analyse tide data
fn = "/Users/ignace/Documents/WETCOAST/Data/Blackwater/tides/Lennert/water_levels_BlackwaterEstuary_MD_USA_20140814-20141029.csv"
df = pd.read_csv(fn)

transformer = Transformer.from_crs("EPSG:26985", "EPSG:4326")

X,Y = np.unique(df['eastCoordinate']), np.unique(df['northCoordinate'])

m = folium.Map(location=transformer.transform(np.mean(X), np.mean(Y)), zoom_start=12)

for i in range(1,6):
    wl = df['measurementValue'][df['fieldSite'] == i]
    t = pd.to_datetime(df['eventDate'][df['fieldSite'] == i])

    x = df['eastCoordinate'][df['fieldSite'] == i].iloc[0]
    y = df['northCoordinate'][df['fieldSite'] == i].iloc[0]

    folium.Marker(
        location=transformer.transform(x, y),
        popup=f'sensor {i}').add_to(m)

#m.show_in_browser()

#%% analyse tide data

s1 = df[df['fieldSite'] == 1]
s1.index = pd.to_datetime(s1['eventDate'])

h = pd.DataFrame()
h.index = s1.index
h['H_NAVD'] = s1['measurementValue']

h['day'] = s1.index.dayofyear
h['halfday'] = 2*(h['day'] - 1) + (h.index.hour - 12)//12
MHHW = h.groupby('day')['H_NAVD'].max().mean()
MHW = h.groupby('halfday')['H_NAVD'].max().mean()

peaks_year_blackwater = h.loc[h.groupby('halfday')['H_NAVD'].idxmax()][['H_NAVD']]

fig, ax = plt.subplots()
ax.plot(h.index, h['H_NAVD'])
ax.scatter(peaks_year_blackwater.index, peaks_year_blackwater.H_NAVD, zorder = 10)

#%% calculate average tide based on Lennert's data
h_avg = []
for peak_time, peak in peaks_year_blackwater.iterrows():
    h_tide = h.loc[peak_time - pd.Timedelta(hours = 6):peak_time + pd.Timedelta(hours = 6)]
    if len(h_tide) == 49:
        h_avg.append(h_tide['H_NAVD'].values - peak['H_NAVD'])
h_avg = np.asarray(h_avg)
h_std = h_avg.std(axis = 0)
h_avg = h_avg.mean(axis = 0)
t = (np.arange(len(h_avg))-24)*.25

fig_avg, ax_avg = plt.subplots()
ax_avg.fill_between(t, h_avg - h_std, h_avg + h_std, alpha = .25)
ax_avg.plot(t,h_avg)

ax_avg.set_xlabel('hours before/after high tide')
ax_avg.set_ylabel('average water level - relative to hwl[m NADP]')

avg_tide = pd.DataFrame({'avg_H': h_avg, 'std_H': h_std}, index = t*3600)
avg_tide.to_csv('/Users/ignace/Documents/WETCOAST/Data/Blackwater/tides/Blackwater_avg_H.csv')
#%% Load Bishopshead

fn = "/Users/ignace/Documents/WETCOAST/Data/Blackwater/tides/CO-OPS_8571421_wl.csv"
raw = pd.read_csv(fn)
raw_time = raw['Date'] + ' ' + raw['Time (GMT)']
raw.index = pd.DatetimeIndex(pd.to_datetime(raw_time)).tz_localize('UTC-05:00')
raw = raw[raw['Verified (ft)'] != '-']
raw['H_NAVD'] = np.asarray(raw['Verified (ft)'], dtype=float)*0.3048
ax.plot(raw.index, raw['H_NAVD'])

h = pd.DataFrame()
h.index = raw.index
h['H_NAVD'] = raw['H_NAVD']
h['day'] = raw.index.dayofyear
h['halfday'] = 2*(h['day'] - 1) + (h.index.hour - 12)//12
MHHW = h.groupby('day')['H_NAVD'].max().mean()
MHW = h.groupby('halfday')['H_NAVD'].max().mean()

peaks_year_bishop = h.loc[h.groupby('halfday')['H_NAVD'].idxmax()][['H_NAVD']]

peaks_year_bishop_sepoct = peaks_year_bishop.loc[(peaks_year_blackwater.index[1] - pd.Timedelta(hours = 2)):(peaks_year_blackwater.index[-1]+pd.Timedelta(hours = 2))]

matched = pd.merge_asof(peaks_year_blackwater, peaks_year_bishop_sepoct, left_index = True, right_index = True, direction = 'nearest')
diff = matched.H_NAVD_x - matched.H_NAVD_y
diff_median = diff.median()
diff_std = diff.std()

#%% correct the HWL data from Bishops Head (full year) to the Blackwater Estuary (only couple of months)
peaks_year_blackwater_est = peaks_year_bishop.copy()
#peaks_year_blackwater_est['H_NAVD'] += diff_median

ax.scatter(peaks_year_blackwater_est.index, peaks_year_blackwater_est.H_NAVD, zorder = 10, c = 'C2')
peaks_year_blackwater_est.to_csv('/Users/ignace/Documents/WETCOAST/Data/Blackwater/tides/Blackwater_est_peaks_2014.csv')

