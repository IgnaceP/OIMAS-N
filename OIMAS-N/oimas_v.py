import numpy as np
import copy
import os
from julia import Main as Julia

Julia.include(f'{os.environ["MARSEDPATH"]}/MARSED.jl')

class OIMAS_N_vectorized(object):
    def __init__(self, n_locations = 10000,
                 n_layers = 100, max_layer_thickness = .5,
                 dt = 1, t = 0,
                 rho_water = 1000, grav = 9.81,
                 rho_min = 2600, rho_om = 1300,
                 E0_min = 0.4, CI_min = 0.2, sigma_ref_min = 'top',
                 E0_om = 0.25, CI_om = 1.0, sigma_ref_om = 'top',
                 Bmax = 2.5, root_to_shoot = 1, turnover = 0.5,
                 gamma = .11, kappa = .11, lamda = .11,
                 Kla0 = .17, Kre0 = .001, mu_la = 1e6, mu_re = 1e6,
                 chi_la = .32, chi_re = .5,
                 f_C = .52, f_litter = 0.18):

        """
        Initialize OIMAS-N model with parameters for layers, biomass, compaction, and carbon decay
        for multiple locations at once. This module was particularly developed to integrate into the demeter modelling framework

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

        :param  n_locations (int): number of locations
        :param  n_layers (int): number of soil layers
        :param  max_layer_thickness (float): maximum thickness of a layer (m)
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

        # ==================================================================================================================================== #
        # 1. Constants (not variable over locations)
        # ==================================================================================================================================== #


        # === general parameters === #
        self.n_locations    = n_locations           # number of locations (for instance model nodes)
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

        # === organic carbon decay === #
        self.mu_la          = mu_la                 # attenuation length-scale with depth for labile decay (m)
        self.mu_re          = mu_re                 # attenuation length-scale with depth for recalcitrant decay (m)
        self.chi_la         = chi_la                # fraction of mortality routed to labile-fast carbon pool
        self.chi_re         = chi_re                # fraction of mortality routed to labile-slow carbon pool
        self.f_C            = f_C                   # carbon fraction of dry biomass (dimensionless)


        # ==================================================================================================================================== #
        # 2. Variables (variable over locations)
        # ==================================================================================================================================== #


        # === biomass evolution === #
        # Bmax           # maximum above-ground biomass (kg m^-2)
        # root_to_shoot  # root:shoot ratio
        # turnover       # turnover rate (year^-1)
        # gamma          # scale depth for below-ground biomass decay (m)
        # kappa          # scale depth for mortality decay profile (m)
        # lamda          # scale depth for belowground decay profile (m)

        # === organic carbon decay === #
        # Kla0           # surface decay constant for labile pool (year^-1)
        # Kre0           # surface decay constant for recalcitrant pool (year^-1)

        # turn them into arrays if the values are scalars
        params = dict(
            Bmax=Bmax, root_to_shoot=root_to_shoot, turnover=turnover,
            gamma=gamma, kappa=kappa, lamda=lamda, Kla0=Kla0, Kre0=Kre0, f_litter=f_litter
        )
        for name, val in params.items():
            if np.isscalar(val):
                val = np.full(self.n_locations, val, dtype = float)
            elif type(val) == list:
                val = np.array(val)
            setattr(self, name, val)

    def initialize_layers(self, init_min_mass, init_om_mass, f_Cla = None, initial_surface = 0.0, set_bbg = False, bbg = None, autochthonuous_fraction=0.0):
        """
        Initialize the layers with mineral and organic masses.

        :param  init_min_mass (float or array): initial mineral mass per layer (kg)
        :param  init_om_mass (float or array): initial organic mass per layer (kg)
        :param  f_Ca (float): portion of initial carbon pool that is labile

        """
        # handle scalar vs array input; check wether given mineral and organic matter are given as a single value, constant for all layers, or as an array, with an initial stratigraphy
        if np.isscalar(init_min_mass):
            self.min_mass   = np.full([self.n_layers, self.n_locations], init_min_mass, dtype = float)
        elif len(init_min_mass.shape) == 1:
            self.min_mass   = np.asarray(np.tile(init_min_mass, (self.n_locations, 1)).T, dtype = float)
        else:
            self.min_mass   = np.array(init_min_mass, dtype = float)

        if np.isscalar(init_om_mass):
            self.om_mass    = np.full([self.n_layers, self.n_locations], init_om_mass, dtype = float)
        elif len(init_om_mass.shape) == 1:
            self.om_mass    = np.asarray(np.tile(init_om_mass, (self.n_locations, 1)).T, dtype = float)
        else:
            self.om_mass    = np.array(init_om_mass, dtype = float)

        if np.isscalar(initial_surface):
            initial_surface    = np.full([self.n_locations], initial_surface)

        # if ratio labile to recalcitrant fraction is undefined, estimate based on chi_la and chi_re
        if f_Cla == None:
            f_Cla            = self.chi_la / (self.chi_re + self.chi_la)

        # initialize Cla and Cre
        C                   = self.f_C * self.om_mass
        self.Cla            = f_Cla * C
        self.Cre            = (1 - f_Cla) * C
        self.Cla_auto       = self.Cla * autochthonuous_fraction
        self.Cre_auto       = self.Cre * autochthonuous_fraction

        # total mass
        self.mass           = self.min_mass + self.om_mass

        # bulk density calculation
        rho_bulk       = (self.min_mass + self.om_mass) / \
                                (self.min_mass / self.rho_min + self.om_mass / self.rho_om)

        # calculation of thickness
        self.thickness      = self.mass / rho_bulk

        # depth of center of layers
        self.d              = np.cumsum(self.thickness, axis = 0) - self.thickness / 2

        # calculate biomass
        self.biomass()

        # set base level
        self.baselevel      = initial_surface - np.sum(self.thickness, axis = 0)

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
        self.sigma_ref_min = self.buoy_weight[0,:] / 2
        self.sigma_ref_om = self.buoy_weight[0,:] / 2

        # initiate compaction
        self.compaction()

        # update base level after first compaction to start from surface at zero
        delta = initial_surface - self.surface
        self.baselevel += delta
        self.z += delta
        self.surface = initial_surface


    def calculate_buoyant_weight(self):

        # dry bulk density
        rho_dry = self.mass / self.thickness

        # mass fractions
        f_om = self.om_mass / self.mass
        f_min = self.min_mass / self.mass

        # mixture solid density (physically consistent harmonic mean)
        rho_s = 1.0 / ((f_om / self.rho_om) + (f_min / self.rho_min))

        # effective unit weight (physically consistent)
        gamma_eff = rho_dry * (rho_s - self.rho_water) / rho_s * self.grav

        gamma_eff[gamma_eff < 0] = 0  # numerical safety

        # Calculate incremental stress contribution per layer
        stress_increment = gamma_eff * self.thickness

        # Calculate stress at the CENTER (midpoint) of each layer
        # This matches where material properties (like rho_dry) are usually measured
        self.buoy_weight = np.cumsum(stress_increment, axis = 0) - (stress_increment / 2.0)

    def compaction(self, iterations = 5):
        """
        calculate the compaction and update the bulk density of each layer

        :param  iterations (int): number of iterations (the buoyant weight is updated each iteration)
        """

        # initiate compaction
        for _ in range(iterations):

            self.calculate_buoyant_weight()
            # void ratio in function of compaction
            self.sigma          = np.maximum(self.buoy_weight, 1e-6)
            E_min               = self.E0_min - self.CI_min * np.log(self.sigma/self.sigma_ref_min)
            E_om                = self.E0_om - self.CI_om * np.log(self.sigma/self.sigma_ref_om)
            E_min               = np.maximum(0, np.minimum(E_min, self.E0_min))
            E_om                = np.maximum(0, np.minimum(E_om, self.E0_om))

            # calculate lump void ratio (!!! E_min and E_om are calculated from one single DBD, otherwise this is not preferred !!!)
            Pom                 = (self.om_mass) / (self.om_mass + self.min_mass)
            Pmin                = 1 - Pom

            self.Pom            = Pom
            self.E_min          = E_min
            self.E_om           = E_om

            self.E              = E_min**(1 - Pom) * E_om**Pom
            self.E              = np.clip(self.E, 0, 15)

            # update the dry bulk density
            rho_solid           = (Pom / self.rho_om + Pmin / self.rho_min) ** -1
            rho_bulk            = rho_solid / (1 + self.E)


        # calculation of tickness
        self.thickness      = self.mass / rho_bulk

        # split layers if they get too thick
        self.split_top_layer()

        # update vertical coordinate, surface level and depths
        self.update_geometry()

        # update buoyant weight
        self.calculate_buoyant_weight()

    def update_geometry(self, surface = None):
        """
        Update surface elevation, layer centers, and z-coordinates.
        """
        self.surface            = np.sum(self.thickness, axis = 0) + self.baselevel
        self.d                  = np.cumsum(self.thickness, axis = 0) - self.thickness / 2
        self.z                  = self.surface - self.d

        if surface is not None:
            self.surface = surface
            self.z = self.surface - self.d
            self.baselevel = self.surface - np.sum(self.thickness, axis = 0)

    def split_top_layer(self):
        """
        Split top layer if it exceeds max_layer thickness.
        Conserves mass, organic matter, and carbon pools.
        """

        if not hasattr(self, "bbg"):
            return

        i = self.thickness[0,:] > self.max_layer

        while i.any():

            self.baselevel[i] = self.baselevel[i] + self.thickness[-1, i]

            self.min_mass[:,i]       = np.concatenate([np.array([self.min_mass[0,i]/2, self.min_mass[0,i]/2]), self.min_mass[1:-1,i]])
            self.om_mass[:,i]        = np.concatenate([np.array([self.om_mass[0,i]/2, self.om_mass[0,i]/2]), self.om_mass[1:-1,i]])
            self.Cla[:,i]            = np.concatenate([np.array([self.Cla[0,i]/2, self.Cla[0,i]/2]), self.Cla[1:-1,i]])
            self.Cre[:,i]            = np.concatenate([np.array([self.Cre[0,i]/2, self.Cre[0,i]/2]), self.Cre[1:-1,i]])
            self.Cla_auto[:,i]       = np.concatenate([np.array([self.Cla_auto[0,i]/2, self.Cla_auto[0,i]/2]), self.Cla_auto[1:-1,i]])
            self.Cre_auto[:,i]       = np.concatenate([np.array([self.Cre_auto[0,i]/2, self.Cre_auto[0,i]/2]), self.Cre_auto[1:-1,i]])
            self.thickness[:,i]      = np.concatenate([np.array([self.thickness[0,i]/2, self.thickness[0,i]/2]), self.thickness[1:-1,i]])
            self.bbg_per_layer[:,i]  = np.concatenate([np.array([self.bbg_per_layer[0,i] / 2]), np.array([self.bbg_per_layer[0,i] / 2]), self.bbg_per_layer[1:-1,i]])

            self.mass[:,i]           = self.min_mass[:,i] + self.om_mass[:,i] + self.bbg_per_layer[:,i]

            i                       = (self.thickness[0,:] > self.max_layer)



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
        bbg_per_layer           = bbg * self.thickness

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
        self.Agb                = Bag
        self.Bbg                = Bbg
        self.bbg                = bbg
        self.bbg_per_layer      = bbg_per_layer
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

        # half of the dead above ground biomass
        self.Cla[0,:]           += self.Agb * self.f_litter * self.chi_la * self.f_C

        # evolution of recalcitrant carbon pool (kg m^-2)
        Cre_evol                = -Kre * self.Cre * self.dt + self.Mbg_int * self.chi_re * self.f_C
        self.Cre                += Cre_evol
        self.Cre                = np.maximum(self.Cre, 0)

        # half of the dead above ground biomass
        self.Cre[0,:]           += self.Agb * self.f_litter * self.chi_re * self.f_C

        # evolution of autochtonous carbon pool (kg m^-2)
        self.Cla_auto           += -Kla * self.Cla_auto * self.dt + self.Mbg_int * self.chi_la * self.f_C
        self.Cre_auto           += -Kre * self.Cre_auto * self.dt + self.Mbg_int * self.chi_re * self.f_C
        self.Cla_auto           = np.maximum(self.Cla_auto, 0)
        self.Cre_auto           = np.maximum(self.Cre_auto, 0)
        self.Cla_auto[0,:]      += self.Agb * self.f_litter * self.chi_la * self.f_C
        self.Cre_auto[0,:]      += self.Agb * self.f_litter * self.chi_re * self.f_C

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
        self.mass               = self.om_mass + self.min_mass + self.bbg_per_layer


        # Recalculate layer properties
        # Call compaction to update:
        #   - bulk density (self.rho_bulk)
        #   - thickness (self.thickness)
        #   - layer depths (self.d)
        #   - buoyant weight (self.buoy_weight)
        #   - the compaction itself
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
        :param dt (int): timestep (seconds) of the MARSED model within one tidal cycle
        :param f_Cla (float): portion of sediment carbon pool that is labile
        :return:
        """
        if rho is None:
            rho = (self.mass / self.thickness)[0,:]

        k = np.asarray(k, dtype = np.float32)
        rho = np.asarray(rho, dtype = np.float32)
        self.surface = np.asarray(self.surface, dtype = np.float32)

        # make sure the arrays are in the right format to match the Julia types
        hwls = np.asarray(hwls, dtype = np.float32)
        avg_tide_t = np.asarray(avg_tide_t, dtype = np.float32)
        avg_tide_h = np.asarray(avg_tide_h, dtype = np.float32)

        tide_times = np.asarray(np.arange(0, 21600, dt) - 21600 / 2, dtype = np.float32)
        ht = np.zeros_like(tide_times)
        C = np.zeros_like(tide_times)

        # call the MARSED model (written in Julia for computational efficiency)
        upd_surface = Julia.marsed_v(hwls, avg_tide_t, avg_tide_h, tide_times, ht, C,
                                     self.surface, rho, k, ws = ws, dt = dt)

        # get the sedimentation in mass (not m)
        sed = (upd_surface - self.surface) * rho

        # split sediment into organic and mineral
        sed_om = sed_om_frac * sed
        sed_min = (1 - sed_om_frac) * sed

        self.sedimentation(sed_om, sed_min, f_Cla = f_Cla)
    def sedimentation(self, sedimentation_om, sedimentation_min, f_Cla = None):
        """
        method to add sediment from the top

        :param sedimentation_om (array): added organic matter through sedimenation (kg m^-2)
        :param sedimentation_min (array): added mineral sediment through sedimenation (kg m^-2)
        :param f_Cla (float): portion of initial carbon pool that is labile
        """

        # update the mass of the top layers
        self.om_mass[0,:]         += sedimentation_om
        self.min_mass[0,:]        += sedimentation_min

        # if ratio labile to recalcitrant fraction is undefined, estimate based on chi_la and chi_re
        if f_Cla is None:
            f_Cla               = self.chi_la

        C                       = sedimentation_om * self.f_C
        self.Cla[0,:]           += C * f_Cla
        self.Cre[0,:]           += C * (1 - f_Cla)

        # update the total mass
        self.mass               = self.om_mass + self.min_mass + self.bbg_per_layer

    def erosion(self, erosion_mass):
        """
        method to remove sediment from the top

        :param erosion (array): removed total mass (kg m^-2)
        """

        if np.isscalar(erosion_mass):
            raise ValueError("erosion_mass must be an array")

        if (erosion_mass < 0).any():
            raise ValueError("erosion_mass must be positive")

        erosion_mass = erosion_mass.astype(float)

        attributes_to_update = ["om_mass", "min_mass",
                                "Cla", "Cre", "Cla_auto","Cre_auto",
                                "thickness", "bbg_per_layer"]
        counter = 0
        while (erosion_mass > 0.001).any():

            # avoid negative erosion at al cost
            erosion_mass = np.maximum(erosion_mass, 0)

            # calculate how much of the erosion contributes to the top layer
            solid_mass_top_layer    = self.min_mass[0,:] + self.om_mass[0,:]
            erosion_top_layer       = np.minimum(erosion_mass, solid_mass_top_layer)
            erosion_mass            -= erosion_top_layer

            # get solid top layer before erosion
            solid_to_keep_top_layer = 1-(erosion_top_layer/solid_mass_top_layer)

            # update the mass of the top layers
            for attribute in attributes_to_update:
                arr = getattr(self, attribute)
                arr[0,:] = arr[0,:] * solid_to_keep_top_layer
                setattr(self, attribute, arr)

            empty_i = (self.om_mass[0,:] < 0.00001) & (self.min_mass[0,:] < 0.00001)

            for attribute in attributes_to_update:
                arr = getattr(self, attribute)
                fv = 1e-10
                if attribute == "min_mass":
                    fv = 1.
                arr[:,empty_i] = np.concatenate((arr[1:,empty_i], np.full(self.n_locations, fv)[empty_i].reshape(1,-1)), axis=0)
                setattr(self, attribute, arr)

            counter += 1
            if counter >= self.n_layers:
                break



        # update the total mass
        self.bbg                = self.bbg_per_layer / self.thickness
        self.mass               = self.om_mass + self.min_mass + self.bbg_per_layer

        self.update_geometry()  # surface, d, z
        self.calculate_buoyant_weight()
        self.compaction()


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

    def get_C_auto(self):
        """
        method tho retrieve dry bulk density
        :return: dry bulk density (kg m^-3)
        """

        return self.Cre_auto + self.Cla_auto

    def copy(self):
        """
        Return a deep copy of the OIMAS_N instance.

        This allows safe branching of simulations without shared state.
        """
        return copy.deepcopy(self)

    def save(self):
        """

        :return:
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)

