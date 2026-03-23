import numpy as np
import scipy
import copy

class OIMAS_N(object):
    def __init__(self, n_layers = 10, max_layer_thickness = .5,
                 dt = 30, t = 0,
                 rho_water = 1000, grav = 9.81,
                 rho_min = 2600, rho_om = 1300,
                 E0_min = 0.4, CI_min = 0.2, sigma_ref_min = 'top',
                 E0_om = 0.25, CI_om = 1.0, sigma_ref_om = 'top',
                 Bmax = 2.5, Dmax = .55, Dmin = 0., D = .55,
                 theta_Bmin = 0, day_peak = 244, phase = 56,
                 ups_Gmax = 0.0138, nu_Gmin = 0,
                 theta_bg = -6.8, Dmbm = 4.8, root_to_shoot = None,
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

        :param  n_layers (int): number of soil layers
        :param max_layer_thickness (float): maximum thickness of a layer (m)
        :param  dt (float): model timestep (days)
        :param  t (float): current time (days)
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
        :param  Dmax (float): maximum marsh platform depth relative to MHHW (m)
        :param  Dmin (float): minimum marsh platform depth relative to MHHW (m)
        :param  D (float): current marsh platform depth relative to MHHW (m)
        :param  theta_Bmin (float): fraction of minimum above-ground biomass relative to peak biomass
        :param  day_peak (int): day of year for peak above-ground biomass
        :param  phase (float): phase shift between biomass accumulation and growth curves (days)
        :param  ups_Gmax (float): maximum above-ground growth rate coefficient (kg m^-2 day^-1 per biomass unit)
        :param  nu_Gmin (float): minimum above-ground growth rate coefficient (kg m^-2 day^-1 per biomass unit)
        :param  theta_bg (float): linear coefficient for root:shoot ratio with depth
        :param  Dmbm (float): vertical distribution constant for below-ground biomass
        :param  gamma (float): scale depth for below-ground biomass decay (m)
        :param  kappa (float): scale depth for below-ground mortality decay profile (m)
        :param  lamda (float): scale depth for below-ground growth decay profile (m)
        :param  Kla0 (float): surface decay constant for labile carbon pool (timestep^-1; likely to be a month)
        :param  Kre0 (float): surface decay constant for recalcitrant carbon pool (timestep^-1; likely to be a month)
        :param  mu_la (float): attenuation length-scale for labile decay (m)
        :param  mu_re (float): attenuation length-scale for recalcitrant decay (m)
        :param  chi_la (float): fraction of root mortality routed to labile carbon pool
        :param  chi_re (float): fraction of root mortality routed to recalcitrant carbon pool
        :param  f_C (float): carbon fraction of dry biomass (dimensionless)

        """

        # === general parameters === #

        self.n_layers       = n_layers              # number of layers
        self.max_layer      = max_layer_thickness   # maximum layer tickness (m)
        self.dt             = dt                    # time step (days)
        self.rho_water      = rho_water             # bulk density of water (kg m^-3)
        self.grav           = grav                  # gravitational acceleration (m s^-2)
        self.t              = t                     # timestep (days)

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
        self.Bmax           = Bmax                  # maximum above-ground biomass at optimal elevation (kg m^-2)
        self.D              = D                     # marsh platform depth relative to MHHW (m)
        self.Dmin           = Dmin                  # elevation lower bound (m)
        self.Dmax           = Dmax                  # elevation upper bound (m)
        self.theta_Bmin     = theta_Bmin            # proportional minimum above-ground biomass at low elevation (fraction of Bp)
        self.day_peak       = day_peak              # peak biomass day (DOY, typically mid-August)
        self.ups_Gmax       = ups_Gmax              # max above-ground growth rate coefficient (kg m^-2 day^-1 per biomass unit)
        self.nu_Gmin        = nu_Gmin               # min above-ground growth coefficient (can be 0)
        self.phase          = phase                 # phase shift between biomass accumulation and growth curves (days)
        self.theta_bg       = theta_bg              # linear coefficient of the relation between the roots:shoots ratio and the depth below MHHW
        self.Dmbm           = Dmbm                  # intercept of root-shoot ratio equation
        self.root_to_shoot  = root_to_shoot         # root:shoot ratio, as alternative for theta_bg and Dmbm
        self.gamma          = gamma                 # scale depth for below-ground biomass decay (m)
        self.kappa          = kappa                 # scale depth for mortality decay profile (m)
        self.lamda          = lamda                 # scale depth for growth decay profile (m)

        # === organic carbon decay === #

        self.Kla0           = Kla0                  # surface decay constant for labile pool (day^-1)
        self.Kre0           = Kre0                  # surface decay constant for recalcitrant pool (day^-1)
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
        self.mass           += self.bbg[:,-1]

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
            rho_bulk            = (self.rho_om * ((self.om_mass + self.bbg[:,-1]) / self.mass) + self.rho_min * (self.min_mass / self.mass)) / (1 + self.E)

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
            top_layer = self.bbg[0:1, :] / 2  # shape (1, n_days)
            self.bbg = np.vstack([top_layer, top_layer, self.bbg[1:, :]])

            self.mass           = self.min_mass + self.om_mass
            self.n_layers       = len(self.thickness)


    def biomass(self):
        """
        calculate the biomass evolution

        !!! timestep of the biomass calculation is daily !!!
        !!! at the end the mortality rate is integrated over the model timestep (likely monthly but can be changed) !!!

        """

        # time step to day of the year
        days                    = np.arange(self.t - self.dt, self.t)
        days[days < 1]          = 365 + days[days < 1]
        days[days > 365]        = (days%365)[days > 365]

        # ========================== #
        # Above-ground vegetation
        # ========================== #

        # peak above-ground biomass (kg m^-2)
        Bp                      = self.Bmax * (self.D - self.Dmin) / (self.Dmax - self.Dmin)
        Bmin                    = self.theta_Bmin * Bp

        # above-ground biomass over time (kg m^-2)
        Bag                     = .5 * (Bmin + Bp + (Bp - Bmin) *
                                    np.cos(2 * np.pi * (days - self.day_peak) / 365))
        Bag_extra_day           = .5 * (Bmin + Bp + (Bp - Bmin) *
                                    np.cos(2 * np.pi * (self.t - self.day_peak) / 365))

        # maximum growth grate (kg m^-2 day^-1)
        Gmax                    = self.ups_Gmax * Bp
        # minimum growth rate (kg m^-2 day^-1)
        Gmin                    = self.nu_Gmin * Bp
        # above-ground growth rate (kg m^-2 day^-1)
        Gag                     = .5 * (Gmin + Gmax + (Gmax - Gmin) *
                                    np.cos(2 * np.pi * (days - self.day_peak + self.phase) / 365))

        # above-ground biomass over time (kg m^-2 day^-1)
        Bag_evol                = np.concatenate(
                                    [Bag[1:] - Bag[:-1], [Bag_extra_day - Bag[-1]]])
        # above-ground mortality rate (kg m^-2 day^-1)
        Mag                     = Bag_evol - Gag

        # set class attributes
        self.Bag                = Bag
        self.Gag                = Gag
        self.Mag                = Mag

        # ========================== #
        # Below-ground vegetation
        # ========================== #

        # root-to-shoot ratio
        if self.root_to_shoot is None:
            root_to_shoot           = self.theta_bg * self.D + self.Dmbm
            Bbg                     = Bag * root_to_shoot
        else:
            root_to_shoot           = self.root_to_shoot(days)
            Bbg                     = Bp * root_to_shoot

        # total below-ground biomass (kg m^-2)

        Bbg_extra_day           = Bag_extra_day * (self.theta_bg * self.D + self.Dmbm)
        # below-ground biomass over time (kg m^-2 day^-1)
        Bbg_evol                = np.concatenate([Bbg[1:] - Bbg[:-1],
                                   [Bbg_extra_day - Bbg[-1]]])
        # below-ground biomass per unit volume at the surface of the marsh (kg m^-3)
        b0                      = Bbg / self.gamma
        # below-ground mortality rate (kg m^-2 day^-1)
        Mbg                     = Mag * (self.theta_bg * self.D + self.Dmbm)
        # below-ground biomass mortality at the surface of the marsh (kg m^-3 day^-1)
        m0                      = Mbg / self.kappa

        # reshape depth array
        depth                   = self.d[:, None]

        # below-ground biomass  per depth interval (kg m^-3)
        bbg                     = b0[None, :] * np.exp(-1 * depth / self.gamma)

        # below-ground biomass mortality per depth interval (kg m^-3 day^-1)
        mbg                     = m0[None, :] * np.exp(-1 * depth / self.kappa)

        # integrate over time (days → timestep; kg m^-3 timestep^-1)
        mbg_time_int            = np.sum(mbg, axis=1)

        # convert from per volume to per layer (kg m^-2 timestep^-1)
        bbg_layer               = bbg * self.thickness[:, None]
        mbg_layer               = mbg_time_int * self.thickness

        # -----------------------------
        # Ensure sum of bbg_layer matches Bbg exactly
        total_bbg_layer         = np.sum(bbg_layer, axis=0)
        missing                 = Bbg - total_bbg_layer
        #bbg_layer[0, :]         += missing
        total_mbg_layer         = np.sum(mbg_layer)
        missing_mbg             = np.sum(Mbg) - total_mbg_layer
        #mbg_layer[0]            += missing_mbg
        # -----------------------------

        # set class attributes
        self.Bbg                = Bbg
        self.Mbg                = Mbg
        self.Mbg_int            = mbg_layer
        self.bbg                = bbg_layer

    def organic_carbon_decay(self):
        """
        calculate the organic carbon decay
        """

        # decay coefficient for labile pool
        Kla                     = self.Kla0 * np.exp(-self.d / self.mu_la)

        # decay coefficient for recalcitrant pool
        Kre                     = self.Kre0 * np.exp(-self.d / self.mu_re)

        # evolution of labile carbon pool (kg m^-2)
        Cla_evol                = (-Kla * self.Cla * self.dt
                                    + -1 * self.Mbg_int * self.chi_la * self.f_C)
        self.Cla                += Cla_evol
        self.Cla                = np.maximum(self.Cla, 0)

        # evolution of recalcitrant carbon pool (kg m^-2)
        Cre_evol                = (-Kre * self.Cre * self.dt
                                    + -1 * self.Mbg_int * self.chi_re * self.f_C)
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
        #print(f'ready with time step: {self.t} days')

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

    def get_units(self):
        """
        Return a dictionary containing the physical units of all model attributes.

        Units follow SI conventions and the formulations used in Mudd (2009),
        Kirwan & Mudd (2012), and Bruns et al. (2025).
        """

        units = {

            # === general parameters ===
            "n_layers": "–",
            "dt": "days",
            "t": "days",
            "rho_water": "kg m^-3",
            "grav": "m s^-2",

            # === material densities ===
            "rho_min": "kg m^-3",
            "rho_om": "kg m^-3",

            # === compaction parameters ===
            "E0_min": "–",
            "CI_min": "–",
            "sigma_ref_min": "kg m^-1 s^-2",
            "E0_om": "–",
            "CI_om": "–",
            "sigma_ref_om": "kg m^-1 s^-2",

            # === layer state variables ===
            "min_mass": "kg m^-2",
            "om_mass": "kg m^-2",
            "mass": "kg m^-2",
            "thickness": "m",
            "d": "m",
            "buoy_weight": "Pa",

            # === biomass parameters ===
            "Bmax": "kg m^-2",
            "D": "m",
            "Dmin": "m",
            "Dmax": "m",
            "theta_Bmin": "–",
            "day_peak": "day of year",
            "phase": "days",
            "ups_Gmax": "day^-1",
            "nu_Gmin": "day^-1",
            "theta_bg": "–",
            "Dmbm": "–",
            "gamma": "m",
            "kappa": "m",
            "lamda": "m",

            # === biomass state variables ===
            "Bag": "kg m^-2",
            "Bbg": "kg m^-2",
            "bbg": "kg m^-3",
            "Gag": "kg m^-2 day^-1",
            "Mag": "kg m^-2 day^-1",
            "Mbg": "kg m^-3 day^-1",
            "Mbg_int": "kg m^-2 timestep^-1",

            # === carbon pools ===
            "Cla": "kg C m^-2",
            "Cre": "kg C m^-2",

            # === carbon decay parameters ===
            "Kla0": "day^-1",
            "Kre0": "day^-1",
            "mu_la": "m",
            "mu_re": "m",
            "chi_la": "–",
            "chi_re": "–",
            "f_C": "–",

            # === carbon / OM fluxes ===
            "om_evol": "kg OM m^-2 timestep^-1",
        }

        return units

    def copy(self):
        """
        Return a deep copy of the OIMAS_N instance.

        This allows safe branching of simulations without shared state.
        """
        return copy.deepcopy(self)




