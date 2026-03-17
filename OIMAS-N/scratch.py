import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress

for zone in ["S10y", "S20y", "S40y"]:

    df = pd.read_csv(f"/Users/ignace/Documents/WETCOAST/Data/Saefthinge/LiDAR/ElevationOverTime_tables/Original_Resolution/{zone}_2m.csv",
        skiprows=0, index_col=0)

    zone_years = int(zone.split('y')[0][1:])
    df = df[[c for c in df.columns if int(c) >= 2025 - zone_years]]

    fig, ax = plt.subplots()
    for i in range(len(df)):
        elev = df.iloc[i]
        years = [int(c) for c in df.columns]

        # fit a linear regression
        slope, intercept, r_value, p_value, std_err = linregress(years, elev)
        ax.plot(years, intercept + slope * np.array(years))
        print(df.index[i], slope, intercept, r_value**2)

        # plot elevation values
        label = f'S{df.index[i]}y: %.3f m/yr' % slope
        ax.scatter(years, elev, label = label)


    ax.legend()
    ax.set_xlabel('year')
    ax.set_ylabel('elevation [m]')

    elev_10y = pd.DataFrame({'id': [f'S{df.index[i]}y' for i in range(8)],
                             'elevation_rate_m/year': [slope for i in range(8)]})
    elev_10y.to_csv(f"/Users/ignace/Documents/WETCOAST/Data/Saefthinge/LiDAR/ElevationOverTime_tables/Original_Resolution/{zone}_elevation_rates.csv")