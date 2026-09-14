import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from OIMAS import OIMAS_N
import os
# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = f"/Users/ignace/Documents/WETCOAST/Data"
DATA_DIR = f"{BASE_DIR}/Blackwater/Mona/"

# =============================================================================
# Data loading
# =============================================================================

def load_soil_carbon():
    """
    Load soil carbon datasets for multiple years.
    """
    
    fn = f"{DATA_DIR}/all_data.csv"
    df_raw = pd.read_csv(fn, skiprows=0, index_col=0)

    df = pd.DataFrame()
    df['depth'] = df_raw['depth_corr']
    df['auger'] = df_raw['Core']

    # diameter auger is 10 cm
    # segments represent a depth of 2.6 cm
    df['DBD']               = df_raw['BD_total_gcm3']*1000
    df['root_percentage_mass'] = 100 * df_raw['BD_root_gcm3'] / df_raw['BD_total_gcm3']
    df['C_percentage']      = df_raw['%C_sediment']
    df['mass_m2']           = 0.026 * df['DBD']
    df['mass']              = 0.026 * np.pi * 0.05**2 * df['DBD']
    df['Cmass_m2']          = df['mass_m2'] * df['C_percentage'] / 100
    df['om_percentage']     = df['C_percentage'] / 0.44
    df['om']                = df['mass_m2'] * df['om_percentage'] / 100
    df['min']               = df['mass_m2'] - df['om']
    df['volume']            = 0.026 * np.pi * 0.05**2

    df.index = df['auger']
    df.drop(columns = 'auger', inplace = True)

    return df
def load_rtk_data():

    """
    Load soil carbon datasets for multiple years.
    """

    fn = f"{DATA_DIR}/raw.csv"
    df_raw = pd.read_csv(fn, skiprows=0, index_col=0, sep=';')
    rtk = df_raw[["Elevation_m"]]
    rtk.index = df_raw['Core']
    rtk = rtk.drop_duplicates()

    sar = df_raw[["Sedimentaccretion_mmy"]]/1000
    sar.columns = ["SAR"]
    sar.index = [c[:-1] for c in df_raw['Core']]
    sar.drop_duplicates(inplace=True)

    return rtk, sar
# =============================================================================
# Plotting
# =============================================================================

def plot_carbon_profiles(axs, soil_data):
    """
    Plot observed carbon profiles for multiple sites.
    """
    for i, site in enumerate(soil_data):
        for _, group in site.groupby("auger"):
            axs[i].scatter(
                group["C_percentage"],
                -1 * group["depth"],
                s=4,
                alpha=1,
                c="white"
            )

        axs[i].set_xlabel("C [%]")

    axs[0].set_ylabel("depth [m]")
    axs[0].set_xlim(0, 25)
    axs[0].set_ylim(-0.9, 0)

def plot_dbd_profiles(axs, soil_data):
    """
    Plot observed DBD profiles for multiple sites.
    """
    for i, site in enumerate(soil_data):
        for _, group in site.groupby("auger"):
            axs[i].scatter(
                group["DBD"],
                -1 * group["depth"],
                s=4,
                alpha=1,
                c="white"
            )


        axs[i].set_xlabel(r"dry bulk density [$g/cm^3$]")

    axs[0].set_title("depth [m]")
    axs[0].set_xlim(0, 1250)
    axs[0].set_ylim(-0.9, 0)

def load_hwls(fn="/Users/ignace/Documents/WETCOAST/Data/Blackwater/tides/Blackwater_est_peaks_2014.csv"):
    return pd.read_csv(fn, skiprows=0, index_col=0, sep = ',', parse_dates=True)

def load_avg_tide(fn="/Users/ignace/Documents/WETCOAST/Data/Blackwater/tides/Blackwater_avg_H.csv"):
  return pd.read_csv(fn, index_col=0)