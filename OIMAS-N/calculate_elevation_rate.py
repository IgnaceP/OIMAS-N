import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.style.use('ip02')

from scipy.stats import linregress

zone = 'S40y'
df = pd.read_csv(f"/Users/ignace/Documents/WETCOAST/Data/Saefthinge/LiDAR/ElevationOverTime_tables/Reprojected_5m_MedianResamplingMethod/{zone}_5m.csv",
    skiprows=0, index_col=0)


df = df[[c for c in df.columns if int(c) >= 2005]]


elev_df = pd.DataFrame({"elevation_rate_m/year": np.empty(len(df))},
                       index=[f'{zone}{df.index[i]}' for i in range(len(df))])

fig, ax = plt.subplots()

for i in range(len(df)):

    # --------- #
    # 2015-2025
    # --------- #

    elev = df.iloc[i]
    years = np.array([int(c) for c in df.columns])

    # remove nans
    years = years[~np.isnan(elev)]
    elev = elev[~np.isnan(elev)]

    # fit a linear regression
    slope, intercept, r_value, p_value, std_err = linregress(years, elev)
    ax.plot(years, intercept + slope * np.array(years),  c = f'C{i}')
    print(df.index[i], slope, p_value)

    # plot elevation values
    label = f'S{df.index[i]}y: %.3f m/yr p-value: %.3f' % (slope, p_value)
    ax.scatter(years, elev, label = label,  c = f'C{i}')

    elev_df["elevation_rate_m/year"][f'{zone}{df.index[i]}'] = slope


ax.legend()
ax.set_xlabel('year')
ax.set_ylabel('elevation [m]')


elev_df.to_csv(f"/Users/ignace/Documents/WETCOAST/Data/Saefthinge/LiDAR/ElevationOverTime_tables/Original_Resolution/{zone}_elevation_rates.csv")