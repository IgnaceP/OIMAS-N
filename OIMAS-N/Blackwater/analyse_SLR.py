import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


slr = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Blackwater/tides/Solomon_island_meantrend.csv', skiprows=4, index_col=0, sep = ',')
slr.head()

msl = slr['Monthly_MSL']
msl_2014 = msl.loc[2014].median()
msl -= msl_2014

fig, ax = plt.subplots()
ax.scatter(slr.index, msl, label = 'SLR')

years_int = np.unique(slr.index)
years_dec = (slr.index + slr.Month/12).values

def sl_exp(t, a, k, b):
    return a * np.exp(k * (t - t0)) + b

t = years_dec          # e.g. 1990, 1991, ...
eta = msl.values         # mean sea level in mm or m
t0 = 2024                    # reference year

popt, pcov = curve_fit(sl_exp, t, eta)
a, k, b = popt

ax.plot(years_dec, sl_exp(years_dec, *popt), 'r-', label = 'fit')
ax.plot(np.arange(1850, 2030), 0.47623*np.exp(0.0161643*(np.arange(1850, 2030) - 2014)) - 0.476, 'b-', label = 'fit')