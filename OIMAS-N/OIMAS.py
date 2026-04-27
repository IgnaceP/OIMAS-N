import numpy as np
import scipy
import os
import copy
from julia import Main as Julia

Julia.include(f'{os.environ["MARSEDPATH"]}/MARSED.jl')

class OIMAS_N(object):
    def __init__(self, n_layers = 10, max_layer_thickness = .5,
                 dt = 1, t = 0,
                 rho_water = 1000, grav = 9.81,
                 rho_min = 2600, rho_om = 1300,
                 E0_min = 0.4, CI_min = 0.2, sigma_ref_min = 'top',
                 E0_om = 0.25, CI_om = 1.0, sigma_ref_om = 'top',
                 Bmax = 2.5, root_to_shoot = 1, turnover = 0.5,
                 gamma = .11, kappa = .11, lamda = .11,
                 Kla0 = .17, Kre0 = .001, mu_la = 1e6, mu_re = 1e6,
                 chi_la = .32, chi_re = .5,
                 f_C = .52):

        """
        Initialize OIMAS-N model with parameters for layers, biomass, compaction, and carbon decay.

        The methods should be called in the right order:
            1) model.biomass()
            2) model.organic_carbon_decay()
            3) model.sedimenation(Som, Smin)
            4) model.update_layers()   # ← compaction happens here

        based on:
        Mudd, S.M., Howell, S.M., Morris, J.T., 2009. Impact of dynamic feedbacks between sedimentation, sea-level rise, and biomass production on near-surface marsh stratigraphy and carbon accumulation. Estuar., Coast. Shelf Sci. 82, 377–389. https://doi.org/10.1016/j.ecss.2009.01.028
        Kirwan, M.L., Mudd, S.M., 2012. Response of salt-marsh carbon accumulation to climate change. Nature 489, 550–553. https://doi.org/10.1038/nature11440
        Bruns, N.E., Noyce, G.L., Kirwan, M.L., 2025. The role of geomorphology in mediating biomass allocation impacts on salt-marsh resilience and carbon accumulation. Estuar., Coast. Shelf Sci. 327, 109549. https://doi.org/10.1016/j.ecss.2025.109549
        Rietl, A. J., Megonigal, J. P., Herbert, E. R. & Kirwan, M. L. Vegetation Type and Decomposition Priming Mediate Brackish Marsh Carbon Accumulation Under Interacting Facets of Global Change. Geophys. Res. Lett. 48, (2021).
        Vahsen, M. L. et al. Cohort Marsh Equilibrium Model (CMEM): History, Mathematics, and Implementation. J. Geophys. Res.: Biogeosciences 129, (2024).

        :param  n_layers (int): number of soil layers
        :param max_layer_thickness (float): maximum thickness of a layer (m)
        :param  dt (integer): model timestep (years)
        :param  t (integer): current time (years)
        :param  rho_water (float): density of water (kg/m³)
        :param  grav (float): gravitational acceleration (m/s²)
        :param  rho_min (float): density of solid mineral fraction (no voids) (kg m^-3)
        :param  rho_om(float): density of organic matter fraction (no voids) (kg m^-3)
        :param  E0_min (float): initial void ratio of mineral fraction
        :param  CI_min (float): compression index of mineral fraction
        :param  sigma_ref_min (float): reference stress for mineral fraction (kg s^-2 m^-1)
        :param  E0_om (float): initial void ratio of organic matter
        :param  CI_om (float): compression index of organic matter fraction
        :param  sigma_ref_om (float): reference stress for organic matter (kg s^-2 m^-1)
        :param  Bmax (float): maximum above-ground biomass at optimal elevation (kg/m²)
        :param  root_to_shoot (float): ratio of root biomass to shoot biomass
        :param  turnover (float): turnover rate of above-ground biomass (year^-1)
        :param  gamma (float): scale depth for below-ground biomass decay (m)
        :param  kappa (float): scale depth for below-ground mortality decay profile (m)
        :param  lambda (float): scale depth for below-ground mortality decay profile (m)
        :param  Kla0 (float): surface decay constant for labile carbon pool (year^-1)
        :param  Kre0 (float): surface decay constant for recalcitrant carbon pool (year^-1)
        :param  mu_la (float): attenuation length-scale for labile decay (m)
        :param  mu_re (float): attenuation length-scale for recalcitrant decay (m)
        :param  chi_la (float): fraction of root mortality routed to labile carbon pool
        :param  chi_re (float): fraction of root mortality routed to recalcitrant carbon pool
        :param  f_C (float): carbon fraction of dry biomass (dimensionless)

        """

        # === general parameters === #

        self.n_layers       = n_layers              # number of layers
        self.max_layer      = max_layer_thickness   # maximum layer tickness (m)
        self.dt             = dt                    # time step (years)
        self.rho_water      = rho_water             # bulk density of water (kg m^-3)
        self.grav           = grav                  # gravitational acceleration (m s^-2)
        self.t              = t                     # timestep (years)
        if not isinstance(self.dt, int) or not isinstance(self.t, int):
            raise ValueError("dt and t must be integers")

        # === bulk densities === #

        self.rho_min        = rho_min               # density of solid mineral fraction (no voids) (kg m^-3)
        self.rho_om         = rho_om                # density of solid organic matter (no voids) (kg m^-3)

        # === compaction === #

        self.E0_min         = E0_min                # void ratio (dimensionless) at reference stress for the mineral fraction
        self.CI_min         = CI_min                # compression index for the mineral fraction
        self.sigma_ref_min  = sigma_ref_min         # reference stress (kg s^-2 m^-1) for the mineral fraction

        self.E0_om          = E0_om                 # void ratio (dimensionless) at reference stress for the organic matter fraction
        self.CI_om          = CI_om                 # compression index for the organic matter fraction
        self.sigma_ref_om   = sigma_ref_om          # reference stress (kg s^-2 m^-1) for the organic matter fraction

        # === biomass evolution === #
        self.Bmax           = Bmax                  # maximum above-ground biomass (kg m^-2)
        self.root_to_shoot  = root_to_shoot         # root:shoot ratio
        self.turnover       = turnover              # turnover rate (year^-1)
        self.gamma          = gamma                 # scale depth for below-ground biomass decay (m)
        self.kappa          = kappa                 # scale depth for mortality decay profile (m)
        self.lamda          = lamda                # scale depth for belowground decay profile (m)

        # === organic carbon decay === #

        self.Kla0           = Kla0                  # surface decay constant for labile pool (year^-1)
        self.Kre0           = Kre0                  # surface decay constant for recalcitrant pool (year^-1)
        self.mu_la          = mu_la                 # attenuation length-scale for labile decay (m)
        self.mu_re          = mu_re                 # attenuation length-scale for recalcitrant decay (m)
        self.chi_la         = chi_la                # fraction of mortality routed to labile-fast carbon pool
        self.chi_re         = chi_re                # fraction of mortality routed to labile-slow carbon pool
        self.f_C            = f_C                   # carbon fraction of dry biomass (dimensionless)

    def initialize_layers(self, init_min_mass, init_om_mass, f_Cla = None, initial_surface = 0.0):
        """
        Initialize the layers with mineral and organic masses.

        :param  init_min_mass (float or array): initial mineral mass per layer (kg)
        :param  init_om_mass (float or array): initial organic mass per layer (kg)
        :param  f_Ca (float): portion of initial carbon pool that is labile

        """
        # handle scalar vs array input; check wether given mineral and organic matter are given as a single value, constant for all layers, or as an array, with an initial stratigraphy
        if np.isscalar(init_min_mass):
            self.min_mass   = np.full(self.n_layers, init_min_mass)
        else:
            self.min_mass   = np.array(init_min_mass)

        if np.isscalar(init_om_mass):
            self.om_mass    = np.full(self.n_layers, init_om_mass)
        else:
            self.om_mass    = np.array(init_om_mass)

        # if ratio labile to recalcitrant fraction is undefined, estimate based on chi_la and chi_re
        if f_Cla == None:
            f_Cla            = self.chi_la / (self.chi_re + self.chi_la)

        # initialize Cla and Cre
        C                   = self.f_C * self.om_mass
        self.Cla            = f_Cla * C
        self.Cre            = (1 - f_Cla) * C

        # total mass
        self.mass           = self.min_mass + self.om_mass

        # bulk density calculation
        rho_bulk       = (self.min_mass + self.om_mass) / \
                                (self.min_mass / self.rho_min + self.om_mass / self.rho_om)

        # calculation of thickness
        self.thickness      = self.mass / rho_bulk

        # depth of center of layers
        self.d              = np.cumsum(self.thickness) - self.thickness / 2

        # calculate biomass
        self.biomass()
        self.mass           += self.bbg

        # set base level
        self.baselevel      = -1 * np.sum(self.thickness)

        # calculate stress without compaction
        self.calculate_buoyant_weight()

        # check whether reference buoyant weight should equal the top layer
        if self.sigma_ref_min == 'top' and self.sigma_ref_om == 'top':
            self.sigma_ref_min = self.buoy_weight[0] / 2
            self.sigma_ref_om = self.buoy_weight[0] / 2
        elif self.sigma_ref_min == 'top' or self.sigma_ref_om == 'top':
            raise ValueError(
                'If reference buoyant weight is set as the buoyant weight of in the center of the top layer, this has to apply to both the mineral and organic matter fraction.')

        # initiate compaction
        self.compaction()

        # freeze sigma_ref_*
        self.sigma_ref_min = self.buoy_weight[0] / 2
        self.sigma_ref_om = self.buoy_weight[0] / 2

        # initiate compaction
        self.compaction()

        # update base level after first compaction to start from surface at zero
        delta = initial_surface - self.surface
        self.baselevel += delta
        self.z += delta
        self.surface = initial_surface

        # initialize age horizons list
        self.age_horizons = [{'t': self.t, 'cum_min_mass': 0.0}]

    def calculate_buoyant_weight(self):

        # dry bulk density
        rho_dry = self.mass / self.thickness

        # mixture solid density
        rho_s = (
                self.rho_om * (self.om_mass / self.mass) +
                self.rho_min * (self.min_mass / self.mass)
        )

        # effective unit weight (physically consistent)
        gamma_eff = rho_dry * (rho_s - self.rho_water) / rho_s * self.grav

        gamma_eff[gamma_eff < 0] = 0  # numerical safety

        # cumulative effective stress
        self.buoy_weight = np.cumsum(gamma_eff * self.thickness)

    def compaction(self, iterations = 5):
        """
        calculate the compaction and update the bulk density of each layer

        :param  iterations (int): number of iterations (the buoyant weight is updated each iteration)
        """

        for _ in range(iterations):

            # void ratio in function of compaction
            self.sigma          = np.maximum(self.buoy_weight, 1e-6)
            E_min               = self.E0_min - self.CI_min * np.log(self.sigma/self.sigma_ref_min)
            E_om                = self.E0_om - self.CI_om * np.log(self.sigma/self.sigma_ref_om)
            E_min               = np.maximum(0, np.minimum(E_min, self.E0_min))
            E_om                = np.maximum(0, np.minimum(E_om, self.E0_om))

            # calculate lump void ratio (!!! E_min and E_om are calculated from one single DBD, otherwise this is not preferred !!!)
            Pom                 = self.om_mass / self.mass
            self.E              = E_min * (1 - Pom) + E_om * Pom

            # update the dry bulk density
            rho_bulk            = (self.rho_om * ((self.om_mass + self.bbg) / self.mass) + self.rho_min * (self.min_mass / self.mass)) / (1 + self.E)

            # calculation of tickness
            self.thickness      = self.mass / rho_bulk

            # split layers if they get too thick
            self.split_top_layer()

            # update vertical coordinate, surface level and depths
            self.update_geometry()

            # update buoyant weight
            self.calculate_buoyant_weight()

    def update_geometry(self):
        """
        Update surface elevation, layer centers, and z-coordinates.
        """
        self.surface            = np.sum(self.thickness) + self.baselevel
        self.d                  = np.cumsum(self.thickness) - self.thickness / 2
        self.z                  = self.surface - self.d

    def split_top_layer(self):
        """
        Split top layer if it exceed max_layer thickness.
        Conserves mass, organic matter, and carbon pools.
        """

        if not hasattr(self, "bbg"):
            return

        while self.thickness[0] > self.max_layer:

            self.min_mass       = np.concatenate([np.array([self.min_mass[0]/2, self.min_mass[0]/2]), self.min_mass[1:]])
            self.om_mass        = np.concatenate([np.array([self.om_mass[0]/2, self.om_mass[0]/2]), self.om_mass[1:]])
            self.Cla            = np.concatenate([np.array([self.Cla[0]/2, self.Cla[0]/2]), self.Cla[1:]])
            self.Cre            = np.concatenate([np.array([self.Cre[0]/2, self.Cre[0]/2]), self.Cre[1:]])
            self.thickness      = np.concatenate([np.array([self.thickness[0]/2, self.thickness[0]/2]), self.thickness[1:]])
            self.bbg            = np.concatenate([np.array([self.bbg[0] / 2]), np.array([self.bbg[0] / 2]), self.bbg[1:]])

            self.mass           = self.min_mass + self.om_mass
            self.n_layers       = len(self.thickness)


    def biomass(self):
        """
        calculate the biomass evolution

        !!! timestep of the biomass calculation is daily !!!
        !!! at the end the mortality rate is integrated over the model timestep !!!

        """




        # ========================== #
        # Above-ground vegetation
        # ========================== #

        # peak above-ground biomass (kg m^-2)
        Bag                      = self.Bmax

        # ========================== #
        # Below-ground vegetation
        # ========================== #

        # root-to-shoot ratio
        Bbg                     = Bag * self.root_to_shoot

        # below-ground biomass per unit volume at the surface of the marsh (kg m^-3)
        b0                      = Bbg / self.gamma
        bbg                     = b0 * np.exp(-1 * self.d / self.lamda)

        # !!! not following Mudd et al. 2009
        # !!! mortality rate of below-ground biomass will be estimated based on the seasonal decrease in below-ground biomass

        # below-ground mortality rate (kg m^-2 timestep^-1)
        Mbg                     = self.dt * Bbg * self.turnover
        m0                      = Mbg / self.kappa

        # below-ground biomass mortality per depth interval (kg m^-3 day^-1)
        mbg                     = m0 * np.exp(-1 * self.d / self.kappa)

        # convert from per volume to per layer (kg m^-2 timestep^-1)
        mbg_layer               = mbg * self.thickness

        # set class attributes
        self.Abg                = Bag
        self.Bbg                = Bbg
        self.bbg                = bbg
        self.Mbg                = Mbg
        self.Mbg_int            = mbg_layer


    def organic_carbon_decay(self):
        """
        calculate the organic carbon decay
        """

        # decay coefficient for labile pool
        Kla                     = self.Kla0 * np.exp(-self.d / self.mu_la)

        # decay coefficient for recalcitrant pool
        Kre                     = self.Kre0 * np.exp(-self.d / self.mu_re)

        # evolution of labile carbon pool (kg m^-2)
        Cla_evol                = -Kla * self.Cla * self.dt + self.Mbg_int * self.chi_la * self.f_C
        self.Cla                += Cla_evol
        self.Cla                = np.maximum(self.Cla, 0)

        # evolution of recalcitrant carbon pool (kg m^-2)
        Cre_evol                = -Kre * self.Cre * self.dt + -1 * self.Mbg_int * self.chi_re * self.f_C
        self.Cre                += Cre_evol
        self.Cre                = np.maximum(self.Cre, 0)

        # evolution of organic matter (kg om m^-2 timestep-1)
        self.om_evol            = (Cre_evol + Cla_evol) / self.f_C


    def update_layers(self):
        """
        method to add the new organic matter (coming from the mortality of the roots) to the layers,
        and update the layers, including calling the compaction method

        """

        # calculate how much of the mortality contributes to the total organic matter mass (kg m^-2 timestep^-1)
        self.om_mass            += self.om_evol
        self.om_mass            = np.maximum(self.om_mass, 0)

        # update the total mass
        self.mass               = self.om_mass + self.min_mass

        # Recalculate layer properties
        # Call compaction to update:
        #   - bulk density (self.rho_bulk)
        #   - thickness (self.thickness)
        #   - layer depths (self.d)
        #   - buoyant weight (self.buoy_weight)
        #   - the compaction itself
        # update buoyant weight
        self.calculate_buoyant_weight()
        self.compaction()

        # update timestep
        self.t                  += self.dt
        #print(f'ready with time step: {self.t} years')


    def marsed(self, hwls, avg_tide_t, avg_tide_h, sed_om_frac = 0.049, ws = 1.1e-4, k = 0.606, dt = 300, f_Cla = None,  rho = None):
        """
        method to calculate sedimentation using the MARSED model
        :param hwls (1D numpy array): high water levels
        :param avg_tide_t (1D numpy array): timings of average high water level
        :param avg_tide_h (1D numpy array): average high water level
        :param sed_om_frac (float): fraction of organic matter that is sedimented
        :param ws (float): sedimentation velocity
        :param k (float): factor to determine the incoming sediment in function of the incoming high water level C0 = k * (HWL - E0)
        :param rho (float): bulk density of deposited sediments (if None, the bulk density of the top layer is used)
        :param dt (int): timestep (minutes) of the MARSED model within one tidal cycle
        :param f_Cla (float): portion of sediment carbon pool that is labile
        :return:
        """
        if rho == None:
            rho = (self.mass / self.thickness)[0]

        # make sure the arrays are in the right format to match the Julia types
        hwls = np.asarray(hwls, dtype = np.float64)
        avg_tide_t = np.asarray(avg_tide_t, dtype = np.float64)
        avg_tide_h = np.asarray(avg_tide_h, dtype = np.float64)

        # call the MARSED model (written in Julia for computational efficiency)
        upd_surface = Julia.marsed(hwls, avg_tide_t, avg_tide_h, E0=self.surface, ws = ws, k = k, dt = dt, rho = rho)
        # get the sedimentation in mass (not m)
        sed = (upd_surface - self.surface) * rho

        # split sediment into organic and mineral
        sed_om = sed_om_frac * sed
        sed_min = (1 - sed_om_frac) * sed

        self.sedimentation(sed_om, sed_min, f_Cla = f_Cla)

    def sedimentation(self, sedimentation_om, sedimentation_min, f_Cla = None):
        """
        method to add sediment from the top

        :param sedimentation_om (float): added organic matter through sedimenation (kg m^-2)
        :param sedimentation_min (float): added mineral sediment through sedimenation (kg m^-2)
        :param f_Ca (float): portion of initial carbon pool that is labile
        """

        # update the mass of the top layers
        self.om_mass[0]         += sedimentation_om
        self.min_mass[0]        += sedimentation_min

        # if ratio labile to recalcitrant fraction is undefined, estimate based on chi_la and chi_re
        if f_Cla == None:
            f_Cla               = self.chi_la / (self.chi_re + self.chi_la)

        C                       = sedimentation_om * self.f_C
        self.Cla[0]             += C * f_Cla
        self.Cre[0]             += C * (1 - f_Cla)

        # update the total mass
        self.mass               = self.om_mass + self.min_mass

        # record new age horizon using cumulative mineral mass (conservative)
        self.age_horizons.append({'t': self.t, 'cum_min_mass': np.sum(self.min_mass)})

    def get_dbd(self):
        """
        method tho retrieve dry bulk density
        :return: dry bulk density (kg m^-3)
        """

        return self.mass / self.thickness

    def get_C(self):
        """
        method tho retrieve dry bulk density
        :return: dry bulk density (kg m^-3)
        """

        return self.Cre + self.Cla

    def get_age_horizons(self):
        cum_min_mass = np.cumsum(self.min_mass[::-1])  # accumulate bottom-up
        result = []
        for h in self.age_horizons:
            idx = min(np.searchsorted(cum_min_mass, h['cum_min_mass']), len(self.z) - 1)
            z = self.z[::-1][idx]  # index into flipped z array
            result.append({'t': h['t'], 'z': z})
        return result

    def copy(self):
        """
        Return a deep copy of the OIMAS_N instance.

        This allows safe branching of simulations without shared state.
        """
        return copy.deepcopy(self)




