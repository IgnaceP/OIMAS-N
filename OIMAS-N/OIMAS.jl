#=
OIMAS:
- Julia version: 
- Author: ignace
- Date: 2026-04-28
=#

Base.@kwdef mutable struct oimas
    n_layers::Int64                 = 100                  # (int): number of soil layers
    max_layer_thickness::Float64    = 0.5                  # (float): maximum thickness of a layer (m)
    dt::Int64                       = 1                    # (int): time step (years)
    t::Float64                      = 0.                   # (float): time (s)
    rho_water::Float64              = 1000.                # (float): density of water (kg/m3)
    grav::Float64                   = 9.81                 # (float): gravitational acceleration (m/s2)
    rho_min::Float64                = 2600.                # (float): density of solid mineral fraction (no voids) (kg m^-3)
    rho_om::Float64                 = 1300.                # (float): density of organic matter fraction (no voids) (kg m^-3)
    E0_min::Float64                 = 0.4                  # (float): reference void ratio of mineral fraction
    CI_min::Float64                 = 0.2                  # (float): compression index of mineral fraction
    E0_om::Float64                  = 0.25                 # (float): reference void ratio of organic matter
    CI_om::Float64                  = 1.                   # (float): compression index of organic matter
    Bmax::Float64                   = 0.                  # (float): maximum above-ground biomass at optimal elevation (kg/m²)
    root_to_shoot::Float64          = 1.                   # (float): ratio of roots to shoots
    turnover::Float64               = 0.5                  # (float): turnover rate of above-ground biomass (year^-1)
    gamma::Float64                  = 0.11                 # (float): scale depth for below-ground biomass decay (m)
    kappa::Float64                  = 0.11                 # (float): scale depth for below-ground mortality decay profile (m)
    lamda::Float64                  = 0.11                 # (float): scale depth for below-ground biomass profile (m)
    Kla::Float64                    = 0.17                 # (float): decay constant for labile carbon pool (year^-1)
    Kre::Float64                    = 0.001                # (float): decay constant for recalcitrant carbon pool (year^-1)
    chi_la::Float64                 = 0.32                 # (float): fraction of root mortality routed to labile carbon pool
    chi_re::Float64                 = 0.5                  # (float): fraction of root mortality routed to recalcitrant carbon pool
    f_C::Float64                    = 0.52                 # (float): carbon fraction of dry biomass (dimensionless)
    buoy_weight_ref::Float64        = 0.                   # (float): reference buoyancy weight (kg)
    baselevel::Float64              = 0.                   # (float): elevation of base of profile (m)
    surface::Float64                = 0.                   # (float): elevation of surface (m)

    min_mass::Vector{Float64}       = Float64[]            # (array): mineral mass per layer (kg)
    om_mass::Vector{Float64}        = Float64[]            # (array): organic matter mass per layer (kg)
    C::Vector{Float64}              = Float64[]            # (array): carbon mass per layer (kg)
    Cla::Vector{Float64}            = Float64[]            # (array): labile carbon mass per layer (kg)
    Cre::Vector{Float64}            = Float64[]            # (array): recalcitrant carbon mass per layer (kg)
    dCladt::Vector{Float64}         = Float64[]            # (array): change in labile carbon per layer (kg m^-2 year^-1)
    dCredt::Vector{Float64}         = Float64[]            # (array): change in recalcitrant carbon per layer (kg m^-2 year^-1)
    mass::Vector{Float64}           = Float64[]            # (array): total mass per layer (kg)
    rho_bulk::Vector{Float64}       = Float64[]            # (array): bulk density per layer (kg/m3)
    thickness::Vector{Float64}      = Float64[]            # (array): thickness per layer (m)
    d::Vector{Float64}              = Float64[]            # (array): depth of center of layers (m)
    buoy_weight::Vector{Float64}    = Float64[]            # (array): buoyancy weight per layer (kg)
    E::Vector{Float64}              = Float64[]            # (array): void ratio per layer
    E_min::Vector{Float64}          = Float64[]            # (array): void ratio of mineral fraction per layer
    E_om::Vector{Float64}           = Float64[]            # (array): void ratio of organic matter fraction per layer
    P_om::Vector{Float64}           = Float64[]            # (array): organic matter fraction per layer
    bbg::Vector{Float64}            = Float64[]            # (array): below-ground biomass per layer (kg m^-2)
    mbg_layer::Vector{Float64}      = Float64[]            # (array): below-ground mortality per layer (kg m^-2 year^-1)
    z::Vector{Float64}              = Float64[]            # (array): z-coordinates (m)
    rho_s::Vector{Float64}          = Float64[]            # (array): mixture solid per layer (kg/m3)
    gamma_eff::Vector{Float64}      = Float64[]            # (array): effective unit weight per layer (kg/m3)
    mbg::Vector{Float64}            = Float64[]            # (array): below-ground mortality (kg m^-2 year^-1)

end

# function to initialze the arrays in oimas
function oimas_init(; n_layers::Int=10, kwargs...)
    obj = oimas(; n_layers=n_layers, kwargs...)
    for field in (:min_mass, :om_mass, :C, :Cla, :Cre,
                  :dCladt, :dCredt, :mass,
                  :rho_bulk, :thickness, :d, :buoy_weight, :E, :bbg, :mbg_layer, :z,
                  :E_min, :E_om, :P_om, :rho_s, :gamma_eff, :mbg)
        setfield!(obj, field, zeros(n_layers))
    end
    return obj
end

function initialize_layers!(oim::oimas, init_min_mass::Vector{Float64}, init_om_mass::Vector{Float64};
     f_Cla=nothing, initial_surface=0.0)
    #=
    Initialize the layers with mineral and organic masses.

    :param  init_min_mass (float or array): initial mineral mass per layer (kg)
    :param  init_om_mass (float or array): initial organic mass per layer (kg)
    :param  f_Ca (float): portion of initial carbon pool that is labile
    :param  initial_surface (float): initial surface elevation (m)

    =#

    # initialize mineral and organic masses
    oim.om_mass            .= init_om_mass
    oim.min_mass           .= init_min_mass

    # initialize surface elevation
    oim.surface         = initial_surface

    # set portion of initial carbon pool that is labile equal to the fraction of the root mortabiility
    if isnothing(f_Cla)
        f_Cla = oim.chi_la / (oim.chi_re + oim.chi_la)
    end

    # initialize Cla and Cre
    @. oim.C               = oim.f_C * oim.om_mass
    @. oim.Cla             = f_Cla * oim.C
    @. oim.Cre             = (1 - f_Cla) * oim.C

    # initialize mass
    @. oim.mass            = oim.min_mass + oim.om_mass

    # bulk density
    @. oim.rho_bulk        = (oim.rho_min * (oim.min_mass / oim.mass)) + (oim.rho_om * ((oim.om_mass + oim.bbg) / oim.mass))

    # calculation of thickness
    @. oim.thickness       = oim.mass / oim.rho_bulk
    replace!(oim.thickness, NaN => 0)

    # depth of center of layers
    cumsum!(oim.d, oim.thickness)
    @. oim.d               -= oim.thickness/2

    # initialize biomass
    biomass!(oim)

    # calculate stress without compaction
    calculate_buoyant_weight!(oim)

    # calculate reference buoyant weight
    oim.buoy_weight_ref = oim.buoy_weight[1]/2

    # update compaction
    compaction!(oim)

    # update reference buoyant weight
    oim.buoy_weight_ref = oim.buoy_weight[1]/2

    # reset baselevel and surface
    oim.surface         = initial_surface
    oim.baselevel       = oim.surface - sum(oim.thickness)

end

function calculate_buoyant_weight!(oim::oimas)
    #=
    Calculate the buoyant weight of the profile.

    :param  oim (object): instance of the oimas type

    =#

    # mixture solid density
    @. oim.rho_s = oim.rho_om * ((oim.om_mass + oim.bbg) / oim.mass) + oim.rho_min * (oim.min_mass / oim.mass)

    # effective unit weight
    @. oim.gamma_eff = oim.thickness * max((oim.mass / oim.thickness) * (oim.rho_s - oim.rho_water) / oim.rho_s * oim.grav, 0.0)

    # buoyant weight
    cumsum!(oim.buoy_weight, oim.gamma_eff)
    clamp!(oim.buoy_weight, 0, Inf)
    replace!(oim.buoy_weight, NaN => 0)

end

function compaction!(oim::oimas; iterations=5)
    #=
    Compaction of the profile.

    :param  oim (object): instance of the oimas type
    :param  iterations (int): number of iterations
    =#

    for _ in 1:iterations

        # void ratio in function of compaction
        @. oim.E_min               = oim.E0_min - oim.CI_min * log(oim.buoy_weight / oim.buoy_weight_ref)
        @. oim.E_om                = oim.E0_om - oim.CI_om * log(oim.buoy_weight / oim.buoy_weight_ref)
        clamp!(oim.E_min, 0, oim.E0_min)
        clamp!(oim.E_om, 0, oim.E0_om)

        # calculate lump void ratio
        @. oim.P_om                = oim.om_mass / oim.mass
        replace!(oim.P_om, NaN => 0)
        @. oim.E                   = oim.P_om * oim.E_om + (1 - oim.P_om) * oim.E_min

        # bulk density
        @. oim.rho_bulk            = ((oim.rho_om * ((oim.om_mass + oim.bbg) / oim.mass)) + (oim.rho_min * (oim.min_mass / oim.mass))) / (1 + oim.E)

        # calculation of thickness
        @. oim.thickness          = oim.mass / oim.rho_bulk
        replace!(oim.thickness, NaN => 0)

        # update vertical coordinate, surface level and depths
        update_geometry!(oim)

        # update buoyant weight
        calculate_buoyant_weight!(oim)
   end

end

function update_geometry!(oim::oimas)
    #=
    Update the vertical coordinate, surface level and depths.

    :param  oim (object): instance of the oimas type
    =#

    # surface level
    oim.surface         = sum(oim.thickness) + oim.baselevel

    # oim.d will first be the cumulative sum of the layer thickness (more memory-efficient)
    cumsum!(oim.d, oim.thickness)

    # then the halfed thickness is substracted to obtain the depth (center of the layer)
    @. oim.d            -= oim.thickness / 2

    # the z-coordinates is the surface minus the depth
    @. oim.z            = oim.surface - oim.d

end

function biomass!(oim::oimas)
    #=
    Calculate the biomass evolution and mortality for autochtonous carbon input

    :param  oim (object): instance of the oimas type
    =#

    # peak above-ground biomass (kg m^-2) & peak below-ground biomass (kg m^-2)
    Bag                     = oim.Bmax
    Bbg                     = Bag * oim.root_to_shoot

    # below-ground biomass per unit volume over the layers (kg m^-3)
    @. oim.bbg              = (Bbg / oim.gamma) * exp(-oim.d / oim.gamma)

    # below-ground mortality rate (kg m^-2 year^-1)
    Mbg                     = oim.dt * (Bbg * oim.turnover)

    # below-ground mortality per unit volume over the layers (kg m^-3 year^-1)
    @. oim.mbg              = (Mbg / oim.kappa) * exp(-oim.d / oim.kappa)

    # below-ground mortality over the layers (kg m^-2 timestep^-1)
    @. oim.mbg_layer        = oim.mbg * oim.thickness * oim.dt

    # add belowground biomass to total mass per layer
    @. oim.mass             = oim.min_mass + oim.om_mass + oim.bbg

end

function sedimentation!(oim::oimas, sedimentation_om::Float64, sedimentation_min::Float64; f_Cla = nothing)
    #=
    Sedimentation of new mass (mineral and organic at the top)

    :param  oim (object): instance of the oimas type
    :param  sedimentation_om (float): sedimentation of organic matter (kg m^-2 year^-1)
    :param  sedimentation_min (float): sedimentation of mineral matter (kg m^-2 year^-1)
    :param  F_Cla (float): portion of sedimentation carbon pool that is labile

    =#

    # move the layers one lower to free up the top one
    circshift!(oim.om_mass, 1)
    circshift!(oim.min_mass, 1)

    # set the new sedimentated mass as the new top layer
    oim.om_mass[1] = sedimentation_om * oim.dt
    oim.min_mass[1] = sedimentation_min * oim.dt

    # set portion of initial carbon pool that is labile equal to the fraction of the root mortabiility
    if isnothing(f_Cla)
        f_Cla = oim.chi_la / (oim.chi_re + oim.chi_la)
    end

    # update the carbon layers
    circshift!(oim.Cla, 1)
    circshift!(oim.Cre, 1)
    circshift!(oim.C, 1)
    oim.Cla[1] = f_Cla * oim.om_mass[1] * oim.f_C
    oim.Cre[1] = (1 - f_Cla) * oim.om_mass[1] * oim.f_C
    oim.C[1] = oim.Cla[1] + oim.Cre[1]

    # update the total mass
    @. oim.mass = oim.min_mass + oim.om_mass + oim.bbg

    # update the baselevel
    oim.baselevel += oim.thickness[end]

end

function decay!(oim::oimas)
    #=
    Organic carbon decay

    :param  oim (object): instance of the oimas type

    =#

    # update the organic carbon layers
    # labile pool
    @. oim.dCladt = - oim.Kla * oim.dt * oim.Cla + oim.mbg_layer * oim.chi_la * oim.f_C
    @. oim.Cla += oim.dCladt
    clamp!(oim.Cla, 0, Inf)

    # recalcitrant pool
    @. oim.dCredt = - oim.Kre * oim.dt * oim.Cre + oim.mbg_layer * oim.chi_re * oim.f_C
    @. oim.Cre += oim.dCredt
    clamp!(oim.Cre, 0, Inf)

    # update the total carbon
    @. oim.C = oim.Cla + oim.Cre

    # update the mass
    @. oim.om_mass += (oim.dCladt + oim.dCredt) / oim.f_C
    clamp!(oim.om_mass, 0, Inf)
    @. oim.mass = oim.min_mass + oim.om_mass + oim.bbg

    # update compaction
    calculate_buoyant_weight!(oim)
    compaction!(oim)

    # update time
    oim.t += 1


end





