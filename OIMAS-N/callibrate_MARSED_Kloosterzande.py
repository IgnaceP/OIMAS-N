import sys
sys.path.append('/Users/ignace/Documents/WETCOAST/model/OIMAS-N')
from tqdm import tqdm

import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ip02')

# surpress warnings
import warnings
warnings.filterwarnings("ignore")

#%% Load data
df2025 = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/Getij/Kloosterzande_2025.csv', index_col = 0, parse_dates=True)[:-2]
df2025['day'] = df2025.index.dayofyear
df2025['H_NAP'] /= 100
df2025['H_TAW'] = df2025['H_NAP'] + 2.33
peak_i, peak_h = scipy.signal.find_peaks(df2025['H_TAW'], height = 4, distance = 6*6)

MHW = np.mean(peak_h['peak_heights'])
MHHW = np.mean(df2025.groupby('day').max().mean()['H_TAW'])
MSL = np.mean(df2025['H_TAW'])

df2025['minutes_to_nearest_peak'] = np.zeros_like(df2025.index, dtype = float)

for i in tqdm(range(len(df2025))):
    low_ind = ((((df2025.index[i] - df2025.index[peak_i]).total_seconds()/60)**2)**0.5).argmin()
    df2025['minutes_to_nearest_peak'][i] = ((df2025.index[i] - df2025.index[peak_i]).total_seconds()/60)[low_ind]

df2025 = df2025.loc[df2025['minutes_to_nearest_peak'] > -375]
df2025 = df2025.loc[df2025['minutes_to_nearest_peak'] < 375]

avg_tidal_wave = df2025.groupby('minutes_to_nearest_peak').mean()['H_TAW']
std_tidal_wave = df2025.groupby('minutes_to_nearest_peak').std()['H_TAW']

fig, ax1 = plt.subplots()
ax1.plot(avg_tidal_wave.index, avg_tidal_wave)
ax1.fill_between(avg_tidal_wave.index, avg_tidal_wave - std_tidal_wave, avg_tidal_wave + std_tidal_wave, alpha = .3)
ax1.set_ylabel('water level [m TAW]')
ax1.set_xlabel('minutes before/after high tide')

df = pd.DataFrame({'avg_H': avg_tidal_wave.values, 'std_H': std_tidal_wave.values}, index = avg_tidal_wave.index*60)
df.to_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/Getij/Kloosterzande_avg_H.csv')

#%% get distribution of HWLs

