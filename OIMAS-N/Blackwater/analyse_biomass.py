import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib
matplotlib.style.use('ip02')

mhwl_2023 = 0.70

#%% Data from Rietl et al.
c3_z = np.asarray([0.28, 0.16, 0.05])
c3_agb = np.asarray([0., .6, 0.])

c4_z = np.asarray([0.45, 0.38, 0.27, 0.12])
c4_agb = np.asarray([0, .5, .7, 0])

c3_coeff = np.polyfit(c3_z, c3_agb, 2)
c4_coeff = np.polyfit(c4_z, c4_agb, 3)

fig, ax = plt.subplots()
#ax.plot(mhwl_2023 - c3_z, c3_agb, 'o', c = 'C0')
#ax.plot(np.arange(0, .5, .01), np.polyval(c3_coeff, np.arange(0, .5, .01)), c = 'C0', label = 'Schoenoplectus americanus (Rietl et al. 2021)')
ax.plot(mhwl_2023 - c4_z, c4_agb, 'o', c = 'C4')
#ax.plot(np.arange(0, .5, .01), np.polyval(c4_coeff, np.arange(0, .5, .01)), c = 'C4', label = 'Spartina patens (Rietl et al. 2021)')
ax.set_xlabel('elevation [m]')
ax.set_ylabel('AGB [kg/m2]')

#%% Data from Mona
fn = '/Users/ignace/Documents/WETCOAST/Data/Blackwater/Mona/all_data.xlsx'
veg = pd.read_excel(fn , sheet_name = 'Vegetation', skiprows=2, index_col=-1)
elev = pd.read_excel(fn , sheet_name = 'Elevation', skiprows=2, index_col=-1)
veg = veg.join(elev['elevation (m)'], how = 'left')
# only consider least degraded
veg = veg.loc[[vi for vi in veg.index.values if vi.startswith('D')]]

for dom_species_i, dom_species in enumerate(np.unique(veg['Dominant vegetation type'])):
    ax.scatter(mhwl_2023 - veg['elevation (m)'][veg['Dominant vegetation type'] == dom_species],
               veg['biomass (g/m²)'][veg['Dominant vegetation type'] == dom_species]/1000,
               ec = 'C' + str(dom_species_i), fc = 'none', label = dom_species)


ax.set_ylim(0,4.5)
#%% Data from Mudd et al. 2009
# for S. alterniflora

Dmax = 0.55
Dmin = 0
Bmax = 2.5
MHHW = 0.396
elev = np.arange(0, 1, 0.01)
Bp = Bmax/(Dmax-Dmin)*(MHHW - elev - Dmin)

#ax.plot(mhwl_2023 - elev, Bp, c = 'C1', label = 'Spartina alterniflora (Mudd et al. 2009)')

#%% Data from MOrris et al. 2013
# for S. alterniflora
biomass = (14.8*100*elev - .157*100*elev**2 + 598)/1000
biomass = lambda x: 1.48 * x - .000157 *(x*100)**2 + .598
#ax.plot(mhwl_2023 - elev, biomass(elev), c = 'C1', ls = '--', label = 'Spartina alterniflora (Morrison et al. 2013)')
ax.legend()

#%% Combine to plot polynomial


z = np.arange(-.4, veg['elevation (m)'].min(), 0.01)
agb = biomass(z)
z = np.concatenate([z, veg['elevation (m)'], [mhwl_2023]])
agb = np.concatenate([agb, veg['biomass (g/m²)']/1000, [0]])

agb = agb[np.argsort(z)]
z = z[np.argsort(z)]

mhwl_2023_depth = mhwl_2023 - z
mhwl_2023_depth = np.concatenate([mhwl_2023_depth, [0]])
agb = np.concatenate([agb, [0]])


est_coeff = np.polyfit(mhwl_2023_depth, agb, 5)
d = np.arange(-0.2,1,0.01)
ax.plot(d, np.polyval(est_coeff,d), c = 'w', label = 'Estimated biomass')

#%% two hyperbolas
def polynom_zerointercept(x, a, b, c, d, e):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x

popt, pcov = curve_fit(polynom_zerointercept, mhwl_2023_depth, agb, p0 = est_coeff[:-1])


d = np.arange(-0.2,1,0.01)
ax.plot(d, polynom_zerointercept(d, *popt), c = 'k', label = 'Estimated biomass')

def veg_weights(d):
    if d < 0.12:
        return 1,0,0
    elif d < 0.16:
        w_am = (d - 0.12)/(0.16 - 0.12)
        w_cyn = 1 - w_am
        return w_cyn, w_am, 0
    elif d < 0.19:
        return 0,1,0
    elif d < 0.23:
        w_alt = (d - 0.19)/(0.23 - 0.19)
        w_am = 1 - w_alt
        return 0, w_am, w_alt
    else:
        return 0,0,1

