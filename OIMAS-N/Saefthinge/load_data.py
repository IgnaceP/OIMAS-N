import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from OIMAS import OIMAS_N
import os
# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = f"/Users/ignace/Documents/WETCOAST/Data"
DATA_DIR = f"{BASE_DIR}/Saefthinge/"
SOIL_CARBON_DIR = f"{DATA_DIR}/soil_carbon"
BIOMASS_DIR = f"{DATA_DIR}/biomass"
RTK_DIR = f"{DATA_DIR}/rtk"
LIDAR_DIR = f"{DATA_DIR}/LiDAR"
TIDES_DIR = f"{DATA_DIR}/tides"

# =============================================================================
# Data loading
# =============================================================================

def load_soil_carbon(years):
    """
    Load soil carbon datasets for multiple years.
    """
    
    res = {}
    
    for y in years:
        fn = f"{SOIL_CARBON_DIR}/S{y}y.csv"
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

        res[y] = df

    return res

def load_rtk_data():
  return pd.read_csv(f'{RTK_DIR}/sample_locations_RTK.csv', index_col=1)[["z_TAW"]]
  
def load_sar_data():
  return pd.read_csv(f'{LIDAR_DIR}/surface_elevation_accumulation.csv', index_col=0)

def load_avg_tide():
  return pd.read_csv(f'{TIDES_DIR}/Kloosterzande_avg_H.csv', index_col=0)

def load_hwls():
  return pd.read_csv(f'{TIDES_DIR}/Kloosterzande_HWLs_1986-2025.csv', index_col=0, skiprows = 1, parse_dates = True)

def load_biomass_data():
    fn_biomass = f"{BIOMASS_DIR}/summary.csv"

    return pd.read_csv(fn_biomass, skiprows=0, index_col=0)
def load_bgb_data():
    """
    Load and preprocess below-ground biomass data.
    """
    fn_bgb = f"{BIOMASS_DIR}/BGB2025.csv"
    bgb = pd.read_csv(fn_bgb, skiprows=1, index_col=1)

    footprint = np.pi * 0.05**2
    bgb["mass_m2"] = bgb["mass"] / footprint / 1000
    bgb = bgb[bgb["primary_plant"] != "bare"]

    bgb["depth"] = [
        (float(d.split("-")[-1][:-2]) - 5) / 100
        for d in bgb["Depth"]
    ]

    bgb['auger'] = bgb.index.copy()

    return bgb

def load_elevation_data(years):
    """
    Load elevation data
    """
    elev_data = {
        y: pd.read_csv(f"{DATA_DIR}/LiDAR/ElevationOverTime_tables/Original_Resolution/S{y}y_elevation_rates.csv",
            skiprows=0, index_col=0)
        for y in years
    }

    return elev_data

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
                -0.01 * group["depth"],
                s=4,
                alpha=0.25,
                c="grey"
            )


        median_C = site.groupby("depth")["C_percentage"].median()
        axs[i].plot(
            median_C,
            -0.01 * median_C.index,
            marker="o",
            c="grey",
            alpha=0.75,
            label="observed C [%]"
        )

        axs[i].set_xlabel("C [%]")

    axs[0].set_ylabel("depth [m]")
    axs[0].set_xlim(0, 9)
    axs[0].set_ylim(-0.9, 0)

def plot_dbd_profiles(axs, soil_data):
    """
    Plot observed DBD profiles for multiple sites.
    """
    for i, site in enumerate(soil_data):
        for _, group in site.groupby("auger"):
            axs[i].scatter(
                1000 * group["DBD"],
                -0.01 * group["depth"],
                s=4,
                alpha=0.25,
                c="grey"
            )

        median_dbd = site.groupby("depth")["DBD"].median()
        axs[i].plot(
            1000 * median_dbd,
            -0.01 * median_dbd.index,
            marker="o",
            c="grey",
            alpha=0.75,
            label="DBD"
        )

        axs[i].set_xlabel(r"dry bulk density [$g/cm^3$]")

    axs[0].set_ylabel("depth [m]")
    axs[0].set_xlim(0, 1950)
    axs[0].set_ylim(-0.9, 0)

def plot_belowground_biomass(axs, biomass_data):
    """
    Plot observed belowground biomass profiles for multiple sites.
    """
    for i, site in enumerate(biomass_data):
        for _, group in site.groupby("auger"):
            axs[i].scatter(
                group["mass_m2"],
                -1 * group["depth"],
                s=4,
                alpha=0.25,
                c="green"
            )
            print(group["mass_m2"])

        median_bgb = site.groupby("depth")["mass_m2"].median()
        axs[i].plot(
            median_bgb,
            -1. * median_bgb.index,
            marker="o",
            c="green",
            alpha=0.75,
            label="Belowground Biomass"
        )

        axs[i].set_xlabel("Belowground Biomass [kg/m^2]")
        axs[i].set_ylabel("")

    axs[0].set_xlim(0, 1.25 * max([max(site["mass_m2"]) for site in biomass_data]))

# =============================================================================
# Model setup
# =============================================================================

def compute_initial_masses(S0y, n_layers):
    """
    Compute initial OM and mineral mass profiles from observations.
    """
    median_mass = S0y.groupby("depth")["mass_m2"].median()
    median_om = S0y.groupby("depth")["om_percentage"].median()

    om_mass = (median_mass * median_om / 100).dropna().values
    min_mass = (median_mass * (100 - median_om) / 100).dropna().values

    om_init = np.zeros(n_layers)
    min_init = np.zeros(n_layers)

    om_init[:len(om_mass)] = om_mass
    om_init[len(om_mass):] = om_mass[-1]

    min_init[:len(min_mass)] = min_mass
    min_init[len(min_mass):] = min_mass[-1]

    return om_init, min_init


