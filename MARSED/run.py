import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from time import time

from MARSED import marsed
from julia import Main as Julia

Julia.include('MARSED.jl')

# load average tidal cycle
df_fn = '/Users/ignace/Documents/WETCOAST/Data/Saefthinge/Getij/Kloosterzande_avg_H.csv'
df = pd.read_csv(df_fn, index_col = 0)

# distribution of high water levels
HWLs_df = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/Getij/Kloosterzande_HWLs_1987-2025.csv', index_col = 0, parse_dates=True)
new_index_1987 = [pd.to_datetime(i) - pd.to_timedelta(365, unit = 'd') for i in HWLs_df.loc['1988-01-01':'1988-04-09'].index]
new_values_1987 = HWLs_df.loc['1988-01-01':'1988-04-09'].values[:,0]
new_index_1986 = [pd.to_datetime(i) - pd.to_timedelta(2*365, unit = 'd') for i in HWLs_df.loc['1988-01-01':'1988-12-31'].index]
new_values_1986 = HWLs_df.loc['1988-01-01':'1988-12-31'].values[:,0]
HWLs_df = pd.concat([pd.Series(new_values_1987, index = new_index_1987), pd.Series(new_values_1986, index = new_index_1986), HWLs_df['H_TAW']]).sort_index()
HWLs = HWLs_df.values

t0 = time()

E = [4.5]
dt = 365*2
julia = False
for t in range(0, len(HWLs), dt):
    if julia:
        hwls = np.asarray(HWLs[t: t + dt], dtype = np.float64)
        avg_tide_t = np.asarray(df.index, dtype = np.float64)
        avg_tide_h = np.asarray(df.avg_H, dtype = np.float64)
        E.append(Julia.marsed(hwls, avg_tide_t, avg_tide_h, E0=E[-1]))
    else:
        E.append(marsed(HWLs[t: t + dt], df.index, df.avg_H, E0 = E[-1]))


print(f'It took {time() - t0} seconds')


fig, ax = plt.subplots(nrows = 1)
years = np.arange(0, len(HWLs), dt)/365
ax.plot(years, E[1:])
plt.show()


