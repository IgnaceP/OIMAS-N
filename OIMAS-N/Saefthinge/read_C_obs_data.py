
import pandas as pd
import numpy as np

def read_observation_data(fn):
    df = pd.read_csv(fn, skiprows=1, index_col=0)
    df['depth'] = [float(d.split('-')[-1][:-2]) - 2.5 for d in df["Depth"]]
    df['auger'] = [i[-1] for i in df.index]

    # using OCC = 0.5208 * SOM - 1.17 (Ouyang & Lee, 2020)
    # diameter auger is 6 cm
    df['mass_m2']           = 0.05 * 1000 * df['DBD']
    df['mass']              = 0.05 * np.pi * 0.03**2 * 1000 * df['DBD']
    df['Cmass_m2']          = df['mass_m2'] * df['C_percentage'] / 100
    df['om_percentage']     = df['C_percentage'] / 0.5208
    df['om']                = df['mass_m2'] * df['om_percentage'] / 100
    df['min']               = df['mass_m2'] - df['om']
    df['volume']            = 0.05 * np.pi * 0.03**2

    return df
