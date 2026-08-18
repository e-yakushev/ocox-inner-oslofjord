"""
OXYgen DEPletion model, OXYDEP targests on the silmplest possible way of parameterization of the oxygen  (DO) fate in changeable redox conditions.
It has a simplified ecosystem, and simulates production of DO due to photosynthesis and consumation of DO for biota respiraion,
OM mineralization, nitrification, and oxidation of reduced specied of S, Mn, Fe, present in suboxic conditions.
For the details of  OxyDEP  implemented here see (Berezina et al, 2022)
Tracers
=======
OXYDEP consists of 6 state variables ( in N-units):
    Phy - all the phototrophic organisms (phytoplankton and bacteria).
    Phy grows due to photosynthesis, loses inorganic matter
    due to respiraion, and loses organic matter in dissolved (DOM) and particulate (POM)
    forms due to metabolism and mortality. Phy growth is limited by irradiance, temperature and NUT availability.
    Het - heterotrophs, can consume Phy and POM,  produce DOM and POM and respirate NUT.
    NUT - represents oxydized forms of nutrients (i.e. NO3 and NO2 for N),
    that doesn't need additional  oxygen for nitrification.
    DOM - is dissolved organic matter. DOM  includes all kinds of labile dissolved organic matter
    and reduced forms of inorganic nutrients (i.e. NH4 and Urea for N).
    POM - is particular organic matter (less labile than DOM). Temperature affects DOM and POM mineralization.
    Oxy - is dissolved oxygen.

Required submodels
==================
* Photosynthetically available radiation: PAR (W/m²)
"""
module OXYDEPModel

export OXYDEP
export bgh_oxydep_boundary_conditions
export oxydep_sediment_forcings

using Oceananigans: fields
using Oceananigans.Units
using Oceananigans.Fields: Field, TracerFields, CenterField, ZeroField
using Oceananigans.Forcings: DiscreteForcing
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.Grids: bottommost_active_node, Center
using Oceananigans.BoundaryConditions:
    fill_halo_regions!,
    ValueBoundaryCondition,
    FieldBoundaryConditions,
    regularize_field_boundary_conditions
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!
using OceanBioME:
    setup_velocity_fields, show_sinking_velocities, Biogeochemistry, ScaleNegativeTracers
using OceanBioME.Light:
    update_TwoBandPhotosyntheticallyActiveRadiation!,
    default_surface_PAR,
    TwoBandPhotosyntheticallyActiveRadiation
using OceanBioME.Sediments: sinking_flux
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, ValueBoundaryCondition, FieldBoundaryConditions

import Adapt: adapt_structure, adapt
import Base: show, summary
import Oceananigans.Biogeochemistry:
    required_biogeochemical_tracers,
    required_biogeochemical_auxiliary_fields,
    biogeochemical_drift_velocity,
    update_biogeochemical_state!
import OceanBioME: redfield, conserved_tracers
import OceanBioME: maximum_sinking_velocity


const Ci_ = false #Main.Ci_  # true: include Ci fields and calculations; false: skip them

""" Surface PAR and turbulent vertical diffusivity based on idealised mixed layer depth """
@inline PAR⁰(x, y, t) =
    60 * (1 - cos((t + 15days) * 2π / 365days)) * (1 / (1 + 0.2 * exp(-((mod(t, 365days) - 200days) / 50days)^2))) + 2

struct OXYDEP{FT,B,W} <: AbstractContinuousFormBiogeochemistry
    # PHY
    initial_photosynthetic_slope::FT # α, 1/(W/m²)/s
    Iopt::FT   # Optimal irradiance (W/m2) =50 (Savchuk, 2002)
    alphaI::FT # initial slope of PI-curve [d-1/(W/m2)] (Wallhead?)
    betaI::FT  # photoinhibition parameter [d-1/(W/m2)] (Wallhead?)                
    gammaD::FT # adaptation to daylength parameter (-)    
    Max_uptake::FT # Maximum nutrient uptake rate d-1
    Knut::FT # Half-saturation constant for an uptake of NUT by PHY for the NUT/PHY ratio (nd) 
    r_phy_nut::FT # Specific respiration rate, (1/d)
    r_phy_pom::FT # Specific rate of Phy mortality, (1/d)
    r_phy_dom::FT # Specific rate of Phy excretion, (1/d)
    # HET
    r_phy_het::FT # Max.spec. rate of grazing of HET on PHY, (1/d)
    Kphy::FT # Half-sat.const.for grazing of HET on PHY for PHY/HET ratio (nd)
    r_pom_het::FT # Max.spec. rate of grazing of HET on POM, (1/d)
    Kpom::FT # Half-sat.const.for grazing of HET on POM for POM/HET ratio (nd)
    Uz::FT # Food absorbency for HET (nd)
    Hz::FT # Ratio between diss. and part. excretes of HET (nd)
    r_het_nut::FT # Specific HET respiration rate (1/d)
    r_het_pom::FT # Specific HET mortality rate (1/d)
    # POM
    r_pom_nut_oxy::FT # Specific rate of POM oxic decay, (1/d)
    r_pom_dom::FT # Specific rate of POM decomposition, (1/d)
    # DOM
    r_dom_nut_oxy::FT # Specific rate of DOM oxic decay, (1/d)
    # O₂
    O2_suboxic::FT    # O2 threshold for oxic/suboxic switch (mmol/m3)
    r_pom_nut_nut::FT # Specific rate of POM denitrification, (1/d)
    r_dom_nut_nut::FT # Specific rate of DOM denitrification, (1/d)
    OtoN::FT # Redfield (138/16) to NO3, (uM(O)/uM(N))
    CtoN::FT # Redfield (106/16) to NO3, (uM(C)/uM(N)) 
    NtoN::FT # Richards denitrification (84.8/16.), (uM(N)/uM(N))
    NtoB::FT # N[uM]/BIOMASS [mg/m3], (uM(N) / mgWW/m3)
    # Ci
    r_ci_degrad::FT # Specific rate of Ci_ degradation, (1/d)
    r_ci_free_phy::FT # Specific rate of Ci_free "uptake" by PHY i.e. biofouling, (1/d) 
    r_ci_food_het::FT # Specific rate of Ci_PHY, Ci_free, Ci_POM uptake by HET, (1/d)
    thr_ci_food_het::FT # Threshold of Ci_PHY, Ci_free, Ci_POM for HET uptake, (mmol/m3)
    optionals::B
    sinking_velocities::W
end

function OXYDEP(grid;
    initial_photosynthetic_slope::FT = 0.1953 / day, # 1/(W/m²)/s
    Iopt::FT = 80.0,     # (W/m2)
    alphaI::FT = 1.8,   # [d-1/(W/m2)]
    betaI::FT = 5.2e-4, # [d-1/(W/m2)]
    gammaD::FT = 0.71,  # (-)
    Max_uptake::FT = 1.85 / day,  # 1/d 2.0 4 5 1.4
    Knut::FT = 0.8,            # (nd)
    r_phy_nut::FT = 0.10 / day, # 1/d
    r_phy_pom::FT = 0.15 / day, # 1/d
    r_phy_dom::FT = 0.17 / day, # 1/d
    r_phy_het::FT = 0.8 / day,  # 1/d 0.4
    Kphy::FT = 0.1,             # (nd) 0.7
    r_pom_het::FT = 0.7 / day,  # 1/d 0.7
    Kpom::FT = 2.0,     # (nd)
    Uz::FT = 0.6,       # (nd)
    Hz::FT = 0.5,       # (nd)
    r_het_nut::FT = 0.15 / day,      # 1/d 0.05
    r_het_pom::FT = 0.10 / day,      # 1/d 0.02
    r_pom_nut_oxy::FT = 0.02 / day,  # 1/d
    r_pom_dom::FT = 0.10 / day,      # 1/d
    r_dom_nut_oxy::FT = 0.05 / day,  # 1/d
    O2_suboxic::FT = 20.0,           # mmol/m3
    r_pom_nut_nut::FT = 0.005 / day, # 1/d
    r_dom_nut_nut::FT = 0.003 / day, # 1/d
    OtoN::FT = 8.625, # (nd)
    CtoN::FT = 6.625, # (nd)
    NtoN::FT = 5.3,   # (nd)
    NtoB::FT = 0.016, # (nd)
    r_ci_degrad::FT =  0.003 / day,  # Specific rate of Ci_ degradation, (1/d) 0.003 (Zhang et al., 2025)
    r_ci_free_phy::FT = 100.0 / day, # Specific rate of Ci_free uptake by PHY, (1/d) 
    r_ci_food_het::FT = 1.1 / day, # Specific rate of Ci_PHY, Ci_free, Ci_POM uptake by HET, (1/d)
    thr_ci_food_het::FT = 0.001, # Threshold of Ci_PHY, Ci_free, Ci_POM for HET uptake, (mmol/m3)
    #------ Optional parameters ------
    surface_photosynthetically_active_radiation = PAR⁰,
    light_attenuation_model::LA = TwoBandPhotosyntheticallyActiveRadiation(;
        grid,
        surface_PAR = surface_photosynthetically_active_radiation,
    ),
    sediment_model::S = nothing,
    TS_forced::Bool = false,
    Chemicals::Bool = false,
    sinking_speeds = Ci_ ?
        (P = 1.0 / day, HET = 4.0 / day, POM = 9.0 / day,
         Ci_PHY = 1.0 / day, Ci_HET = 4.0 / day, Ci_POM = 9.0 / day) :
        (P = 1.0 / day, HET = 4.0 / day, POM = 9.0 / day),
    open_bottom::Bool = true,
    scale_negatives = true,
    particles::P = nothing,
    modifiers::M = nothing,
) where {FT,LA,S,P,M}

    sinking_velocities = setup_velocity_fields(sinking_speeds, grid, open_bottom)
    optionals = Val((TS_forced, Chemicals))

    underlying_biogeochemistry = OXYDEP(
        initial_photosynthetic_slope,
        Iopt,
        alphaI,
        betaI,
        gammaD,
        Max_uptake,
        Knut,
        r_phy_nut,
        r_phy_pom,
        r_phy_dom,
        r_phy_het,
        Kphy,
        r_pom_het,
        Kpom,
        Uz,
        Hz,
        r_het_nut,
        r_het_pom,
        r_pom_nut_oxy,
        r_pom_dom,
        r_dom_nut_oxy,
        O2_suboxic,
        r_pom_nut_nut,
        r_dom_nut_nut,
        OtoN,
        CtoN,
        NtoN,
        NtoB,
        r_ci_degrad,
        r_ci_free_phy,
        r_ci_food_het,
        thr_ci_food_het,
        optionals,
        sinking_velocities,
    )

    if scale_negatives
        scaler = ScaleNegativeTracers(underlying_biogeochemistry, grid)
        modifiers = isnothing(modifiers) ? scaler : (modifiers..., scaler)
    end

    return Biogeochemistry(
        underlying_biogeochemistry;
        light_attenuation = light_attenuation_model,
        sediment = sediment_model,
        particles,
        modifiers,
    )
end

if Ci_
    required_biogeochemical_tracers(::OXYDEP{<:Any,<:Val{(false, false)},<:Any}) =
        (:NUT, :P, :HET, :POM, :DOM, :O₂, :T, :Ci_free, :Ci_PHY, :Ci_HET, :Ci_POM, :Ci_DOM)
else
required_biogeochemical_tracers(::OXYDEP{<:Any,<:Val{(false, false)},<:Any}) =
    (:NUT, :P, :HET, :POM, :DOM, :O₂, :T)
end
required_biogeochemical_auxiliary_fields(::OXYDEP{<:Any,<:Val{(false, false)},<:Any}) = (:PAR,)

@inline function biogeochemical_drift_velocity(bgc::OXYDEP, ::Val{tracer_name}) where {tracer_name}
    if tracer_name in keys(bgc.sinking_velocities)
        return (u = ZeroField(), v = ZeroField(), w = bgc.sinking_velocities[tracer_name])
    else
        return (u = ZeroField(), v = ZeroField(), w = ZeroField())
    end
end

@inline maximum_sinking_velocity(bgc::OXYDEP) = maximum(abs, bgc.sinking_velocities.POM.w)

adapt_structure(to, oxydep::OXYDEP) = OXYDEP(
    adapt(to, oxydep.initial_photosynthetic_slope),
    adapt(to, oxydep.Iopt),
    adapt(to, oxydep.alphaI),
    adapt(to, oxydep.betaI),
    adapt(to, oxydep.gammaD),
    adapt(to, oxydep.Max_uptake),
    adapt(to, oxydep.Knut),
    adapt(to, oxydep.r_phy_nut),
    adapt(to, oxydep.r_phy_pom),
    adapt(to, oxydep.r_phy_dom),
    adapt(to, oxydep.r_phy_het),
    adapt(to, oxydep.Kphy),
    adapt(to, oxydep.r_pom_het),
    adapt(to, oxydep.Kpom),
    adapt(to, oxydep.Uz),
    adapt(to, oxydep.Hz),
    adapt(to, oxydep.r_het_nut),
    adapt(to, oxydep.r_het_pom),
    adapt(to, oxydep.r_pom_nut_oxy),
    adapt(to, oxydep.r_pom_dom),
    adapt(to, oxydep.r_dom_nut_oxy),
    adapt(to, oxydep.O2_suboxic),
    adapt(to, oxydep.r_pom_nut_nut),
    adapt(to, oxydep.r_dom_nut_nut),
    adapt(to, oxydep.OtoN),
    adapt(to, oxydep.CtoN),
    adapt(to, oxydep.NtoN),
    adapt(to, oxydep.NtoB),
    adapt(to, oxydep.r_ci_degrad),
    adapt(to, oxydep.r_ci_free_phy),
    adapt(to, oxydep.r_ci_food_het),
    adapt(to, oxydep.thr_ci_food_het),
    adapt(to, oxydep.optionals),
    adapt(to, oxydep.sinking_velocities),
)


"""
OxyDep basic biogeochemical transformations between NUT, P, HET, DOM, POM, O2
"""
# Limiting equations and switches
@inline yy(consta, value) = value^2 / (value^2 + consta^2)   #This is a squared Michaelis-Menten type of limiter
@inline F_ox(conc, threshold) = (0.5 + 0.5 * tanh(conc - threshold))
@inline F_subox(conc, threshold) = (0.5 - 0.5 * tanh(conc - threshold))

# P
@inline LimLight(PAR, Iopt) = PAR / Iopt * exp(1.0 - PAR / Iopt)  #!Dependence of P growth on Light (Steel)
@inline LimN(Knut, NUT, P) = yy(Knut, NUT / max(0.0001, P)) #!Dependence of P growth on NUT
@inline Q₁₀(T) = 1.88^(T / 10) # T in °C  # inital for NPZD
#@inline LimT(T) = max(0., 2^((T-10.0)/10.) - 2^((T-32.)/3.)) # ERSEM
# = q10^((T-t_upt_min)/10)-q10^((T-t_upt_max)/3):  q10=2. !Coefficient for uptake rate dependence on t
# t_upt_min=10. !Low  t limit for uptake rate dependence on t; t_upt_max=32 !High t limit for uptake rate dependence on t
@inline LimT(T) = exp(0.0663 * (T - 0.0)) #for Arctic (Moore et al.,2002; Jin et al.,2008) 
# = exp(temp_aug_rate*(T-t_0)):  t_0= 0. !reference temperature temp_aug_rate = 0.0663 !temperature augmentation rate
#@inline light_limitation(PAR, α, Max_uptake) = α * PAR / sqrt(Max_uptake ^ 2 + α ^ 2 * PAR ^ 2)

#@inline GrowthPhy(Max_uptake,PAR,α,T,Knut,NUT,P,Iopt) = Max_uptake*LimT(T)*LimN(Knut,NUT,P)*light_limitation(PAR,α,Max_uptake)*P*Iopt/Iopt
@inline GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) =
    Max_uptake * LimT(T) * LimN(Knut, NUT, P) * LimLight(PAR, Iopt) * α / α
@inline RespPhy(r_phy_nut, P) = r_phy_nut * P
@inline MortPhy(r_phy_pom, P) = r_phy_pom * P
@inline ExcrPhy(r_phy_dom, P) = r_phy_dom * P

# HET
@inline GrazPhy(r_phy_het, Kphy, P, HET) =
    r_phy_het * yy(Kphy, max(0.0, P - 0.01) / max(0.0001, HET)) * HET
@inline GrazPOM(r_pom_het, Kpom, POM, HET) =
    r_pom_het * yy(Kpom, max(0.0, POM - 0.01) / max(0.0001, HET)) * HET
@inline RespHet(r_het_nut, HET) = r_het_nut * HET
@inline MortHet(r_het_pom, HET, O₂, O2_suboxic) =
    (r_het_pom + F_subox(O₂, O2_suboxic) * 0.01 * r_het_pom) * HET

# POM
@inline POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) = 
    O₂ < 0.1 ? zero(O₂) : r_pom_nut_oxy * POM * F_ox(O₂, O2_suboxic)
@inline POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT) =
    NUT < 0.05 ? zero(NUT) : r_pom_nut_nut * POM * F_subox(O₂, O2_suboxic)
#! depends on NUT (NO3+NO2) and DOM (NH4+Urea+"real"DON) ! depends on T ! stops at NUT<0.01 
@inline Autolys(r_pom_dom, POM) = r_pom_dom * POM

# DOM
@inline DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) = 
    O₂ < 0.1 ? zero(O₂) : r_dom_nut_oxy * DOM * F_ox(O₂, O2_suboxic)
@inline DOM_decay_denitr(r_dom_nut_nut, DOM, O₂, O2_suboxic, NUT) =
    NUT < 0.05 ? zero(NUT) : r_dom_nut_nut * DOM * F_subox(O₂, O2_suboxic)
#! depends on NUT (NO3+NO2) and DOM (NH4+Urea+"real"DON) ! depends on T ! stops at NUT<0.01 

# O₂

# Ci_
if Ci_
@inline Ci_phy_degrad(r_ci_degrad, Ci_PHY) = r_ci_degrad * Ci_PHY
@inline Ci_het_degrad(r_ci_degrad, Ci_HET) = r_ci_degrad * Ci_HET
@inline Ci_pom_degrad(r_ci_degrad, Ci_POM) = r_ci_degrad * Ci_POM
@inline Ci_dom_degrad(r_ci_degrad, Ci_DOM) = r_ci_degrad * Ci_DOM
@inline Ci_free_phy(r_ci_free_phy, Max_uptake, PAR, α, T, Knut, NUT, P, Iopt, Ci_free) = 
    Ci_free < 1e-6 ? zero(Ci_free) :
    r_ci_free_phy * GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt)
@inline Ci_free_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_free, thr_ci_food_het) = 
    r_ci_food_het * Uz * GrazPhy(r_phy_het, Kphy, P, HET) * yy(thr_ci_food_het, Ci_free)
@inline Ci_phy_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_PHY, thr_ci_food_het) = 
    r_ci_food_het * Uz * GrazPhy(r_phy_het, Kphy, P, HET) * yy(thr_ci_food_het, Ci_PHY)
@inline Ci_pom_het(r_ci_food_het, Uz, r_pom_het, Kpom, POM, HET, Ci_POM, thr_ci_food_het) = 
    r_ci_food_het * Uz * GrazPOM(r_pom_het, Kpom, POM, HET) * yy(thr_ci_food_het, Ci_POM)    
@inline Ci_het_pom(r_het_pom, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
     Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, O₂, O2_suboxic, thr_ci_food_het) = 
    HET < 1e-6 ? zero(HET) :
    (Ci_free_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_free, thr_ci_food_het) +
     Ci_phy_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_PHY, thr_ci_food_het) +
     Ci_pom_het(r_ci_food_het, Uz, r_pom_het, Kpom, POM, HET, Ci_POM, thr_ci_food_het)) /
     Uz * (1 - Uz) * (1 - Hz) +
     Ci_HET * MortHet(r_het_pom, HET, O₂, O2_suboxic) / HET   

@inline Ci_het_dom(r_het_nut, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
     Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, thr_ci_food_het) = 
    HET < 1e-6 ? zero(HET) :
    (Ci_free_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_free, thr_ci_food_het) +
     Ci_phy_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_PHY, thr_ci_food_het) +
     Ci_pom_het(r_ci_food_het, Uz, r_pom_het, Kpom, POM, HET, Ci_POM, thr_ci_food_het)) /
     Uz * (1 - Uz) * Hz +
     Ci_HET * RespHet(r_het_nut, HET) / HET

@inline Ci_pom_dom( r_pom_nut_oxy, r_pom_nut_nut, r_pom_dom, O2_suboxic, NUT, POM, O₂, Ci_POM) = 
     POM < 1e-6 ? zero(POM) :
     Ci_POM * (POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) +
     POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT) + 
     Autolys(r_pom_dom, POM)) /POM
@inline Ci_phy_dom(r_phy_dom, P, Ci_PHY) = 
     P < 1e-6 ? zero(P) :
     Ci_PHY * ExcrPhy(r_phy_dom, P) / P
@inline Ci_phy_pom(r_phy_pom, P, Ci_PHY) = 
     P < 1e-6 ? zero(P) :
     Ci_PHY * MortPhy(r_phy_pom, P) / P
end # if Ci_
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

if Ci_

@inline function (bgc::OXYDEP)(::Val{:NUT},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    r_phy_nut = bgc.r_phy_nut
    r_het_nut = bgc.r_het_nut
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_dom_nut_oxy = bgc.r_dom_nut_oxy
    NtoN = bgc.NtoN
    r_pom_nut_nut = bgc.r_pom_nut_nut
    O2_suboxic = bgc.O2_suboxic
    r_dom_nut_nut = bgc.r_dom_nut_nut
    Iopt = bgc.Iopt

    return (
        RespPhy(r_phy_nut, P) +
        RespHet(r_het_nut, HET) +
        DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) +
        POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) - 
        GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) -
        NtoN * (
            POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT) +
            DOM_decay_denitr(r_dom_nut_nut, DOM, O₂, O2_suboxic, NUT)
        )
    )
    # Denitrification of POM and DOM leads to decrease of NUT (i.e. NOx)
end

@inline function (bgc::OXYDEP)(::Val{:P},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_phy_nut = bgc.r_phy_nut
    r_phy_pom = bgc.r_phy_pom
    r_phy_dom = bgc.r_phy_dom
    Iopt = bgc.Iopt

    return (
        GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) -
        GrazPhy(r_phy_het, Kphy, P, HET) - RespPhy(r_phy_nut, P) - MortPhy(r_phy_pom, P) -
        ExcrPhy(r_phy_dom, P)
    )
end

@inline function (bgc::OXYDEP)(::Val{:HET},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    r_het_nut = bgc.r_het_nut
    r_het_pom = bgc.r_het_pom
    Uz = bgc.Uz
    O2_suboxic = bgc.O2_suboxic

    return (
        Uz * (GrazPhy(r_phy_het, Kphy, P, HET) + GrazPOM(r_pom_het, Kpom, POM, HET)) -
        MortHet(r_het_pom, HET, O₂, O2_suboxic) - RespHet(r_het_nut, HET)
    )
end

@inline function (bgc::OXYDEP)(::Val{:POM},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_phy_pom = bgc.r_phy_pom
    r_het_pom = bgc.r_het_pom
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_pom_dom = bgc.r_pom_dom
    r_pom_nut_nut = bgc.r_pom_nut_nut
    O2_suboxic = bgc.O2_suboxic

    return (
        (1.0 - Uz) *
        (1.0 - Hz) *
        (GrazPhy(r_phy_het, Kphy, P, HET) + GrazPOM(r_pom_het, Kpom, POM, HET)) +
        MortPhy(r_phy_pom, P) +
        MortHet(r_het_pom, HET, O₂, O2_suboxic) - 
        POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) -
        Autolys(r_pom_dom, POM) - GrazPOM(r_pom_het, Kpom, POM, HET) -
        POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT)
    )
end

@inline function (bgc::OXYDEP)(::Val{:DOM},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_phy_dom = bgc.r_phy_dom
    r_dom_nut_oxy = bgc.r_dom_nut_oxy
    r_pom_dom = bgc.r_pom_dom
    r_pom_nut_nut = bgc.r_pom_nut_nut
    O2_suboxic = bgc.O2_suboxic

    return (
        (1.0 - Uz) *
        Hz *
        (GrazPhy(r_phy_het, Kphy, P, HET) + GrazPOM(r_pom_het, Kpom, POM, HET)) +
        ExcrPhy(r_phy_dom, P) - 
        DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic)  +
        Autolys(r_pom_dom, POM) +
        POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT)
    )
    # Denitrification of "real DOM" into NH4 (DOM_decay_denitr) will not change state variable DOM
end

@inline function (bgc::OXYDEP)(::Val{:O₂},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    r_phy_nut = bgc.r_phy_nut
    r_het_nut = bgc.r_het_nut
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_dom_nut_oxy = bgc.r_dom_nut_oxy
    OtoN = bgc.OtoN
    O2_suboxic = bgc.O2_suboxic
    Iopt = bgc.Iopt

    return (
        -OtoN * (
            RespPhy(r_phy_nut, P) +
            RespHet(r_het_nut, HET) +
            DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) +
            POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) -
            GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) + 
            DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) * 
            (F_subox(O₂, O2_suboxic))
        )
    )
    # (POM_decay_denitr + DOM_decay_denitr) & !denitrification doesn't change oxygen
    # (DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) *(F_subox) !additional consumption of O₂ due to oxidation of reduced froms of S,Mn,Fe etc.
    # in suboxic conditions (F_subox) equals consumption for NH4 oxidation (Yakushev et al, 2008)

end

else # !Ci_

@inline function (bgc::OXYDEP)(::Val{:NUT},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    r_phy_nut = bgc.r_phy_nut
    r_het_nut = bgc.r_het_nut
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_dom_nut_oxy = bgc.r_dom_nut_oxy
    NtoN = bgc.NtoN
    r_pom_nut_nut = bgc.r_pom_nut_nut
    O2_suboxic = bgc.O2_suboxic
    r_dom_nut_nut = bgc.r_dom_nut_nut
    Iopt = bgc.Iopt

    return (
        RespPhy(r_phy_nut, P) +
        RespHet(r_het_nut, HET) +
        DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) +
        POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) - 
        GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) -
        NtoN * (
            POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT) +
            DOM_decay_denitr(r_dom_nut_nut, DOM, O₂, O2_suboxic, NUT)
        )
    )
    # Denitrification of POM and DOM leads to decrease of NUT (i.e. NOx)
end

@inline function (bgc::OXYDEP)(::Val{:P},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_phy_nut = bgc.r_phy_nut
    r_phy_pom = bgc.r_phy_pom
    r_phy_dom = bgc.r_phy_dom
    Iopt = bgc.Iopt

    return (
        GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) -
        GrazPhy(r_phy_het, Kphy, P, HET) - RespPhy(r_phy_nut, P) - MortPhy(r_phy_pom, P) -
        ExcrPhy(r_phy_dom, P)
    )
end

@inline function (bgc::OXYDEP)(::Val{:HET},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    r_het_nut = bgc.r_het_nut
    r_het_pom = bgc.r_het_pom
    Uz = bgc.Uz
    O2_suboxic = bgc.O2_suboxic

    return (
        Uz * (GrazPhy(r_phy_het, Kphy, P, HET) + GrazPOM(r_pom_het, Kpom, POM, HET)) -
        MortHet(r_het_pom, HET, O₂, O2_suboxic) - RespHet(r_het_nut, HET)
    )
end

@inline function (bgc::OXYDEP)(::Val{:POM},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_phy_pom = bgc.r_phy_pom
    r_het_pom = bgc.r_het_pom
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_pom_dom = bgc.r_pom_dom
    r_pom_nut_nut = bgc.r_pom_nut_nut
    O2_suboxic = bgc.O2_suboxic

    return (
        (1.0 - Uz) *
        (1.0 - Hz) *
        (GrazPhy(r_phy_het, Kphy, P, HET) + GrazPOM(r_pom_het, Kpom, POM, HET)) +
        MortPhy(r_phy_pom, P) +
        MortHet(r_het_pom, HET, O₂, O2_suboxic) - 
        POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) -
        Autolys(r_pom_dom, POM) - GrazPOM(r_pom_het, Kpom, POM, HET) -
        POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT)
    )
end

@inline function (bgc::OXYDEP)(::Val{:DOM},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_phy_dom = bgc.r_phy_dom
    r_dom_nut_oxy = bgc.r_dom_nut_oxy
    r_pom_dom = bgc.r_pom_dom
    r_pom_nut_nut = bgc.r_pom_nut_nut
    O2_suboxic = bgc.O2_suboxic

    return (
        (1.0 - Uz) *
        Hz *
        (GrazPhy(r_phy_het, Kphy, P, HET) + GrazPOM(r_pom_het, Kpom, POM, HET)) +
        ExcrPhy(r_phy_dom, P) - 
        DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) +
        Autolys(r_pom_dom, POM) +
        POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT)
    )
    # Denitrification of "real DOM" into NH4 (DOM_decay_denitr) will not change state variable DOM
end

@inline function (bgc::OXYDEP)(::Val{:O₂},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    r_phy_nut = bgc.r_phy_nut
    r_het_nut = bgc.r_het_nut
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_dom_nut_oxy = bgc.r_dom_nut_oxy
    OtoN = bgc.OtoN
    O2_suboxic = bgc.O2_suboxic
    Iopt = bgc.Iopt

    return (
        -OtoN * (
            RespPhy(r_phy_nut, P) +
            RespHet(r_het_nut, HET) +
            DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) +
            POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) -
            GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) # due to OM production and decay in normoxia
            +
            DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) * F_subox(O₂, 0.5 * O2_suboxic)
            )
        )
    # (POM_decay_denitr + DOM_decay_denitr) & !denitrification doesn't change oxygen
    # (DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) *(F_subox) !additional 
    # consumption of O₂ due to oxidation of reduced froms of S,Mn,Fe etc.
    # In suboxic conditions (F_subox) equals consumption for NH4 oxidation (Yakushev et al, 2008)

end

end # if Ci_ (tendency functions)

####################################################################
# Ci_(i) transformations
####################################################################
if Ci_

@inline function (bgc::OXYDEP)(::Val{:Ci_free},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)
    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    Iopt = bgc.Iopt
    r_ci_free_phy = bgc.r_ci_free_phy
    r_ci_food_het = bgc.r_ci_food_het
    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    Uz = bgc.Uz
    thr_ci_food_het = bgc.thr_ci_food_het
    return (
        - Ci_free_phy(r_ci_free_phy, Max_uptake, PAR, α, T, Knut, NUT, P, Iopt, Ci_free)
        - Ci_free_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_free, thr_ci_food_het)
    )
end

@inline function (bgc::OXYDEP)(::Val{:Ci_PHY},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)
    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    Iopt = bgc.Iopt
    r_ci_free_phy = bgc.r_ci_free_phy
    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    Uz = bgc.Uz
    r_ci_food_het = bgc.r_ci_food_het
    r_phy_dom = bgc.r_phy_dom
    r_phy_pom = bgc.r_phy_pom
    thr_ci_food_het = bgc.thr_ci_food_het
    r_ci_degrad = bgc.r_ci_degrad
    return (
          Ci_free_phy(r_ci_free_phy, Max_uptake, PAR, α, T, Knut, NUT, P, Iopt, Ci_free)
        - Ci_phy_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_PHY, thr_ci_food_het)
        - Ci_phy_pom(r_phy_pom, P, Ci_PHY) 
        - Ci_phy_dom(r_phy_dom, P, Ci_PHY)     
        - Ci_phy_degrad(r_ci_degrad, Ci_PHY)
        )
end

@inline function (bgc::OXYDEP)(::Val{:Ci_HET},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)
    r_phy_het = bgc.r_phy_het
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    r_ci_food_het = bgc.r_ci_food_het
    thr_ci_food_het = bgc.thr_ci_food_het
    r_het_pom = bgc.r_het_pom
    r_phy_het = bgc.r_phy_het
    O2_suboxic = bgc.O2_suboxic
    Kphy = bgc.Kphy
    Kpom = bgc.Kpom
    r_het_nut = bgc.r_het_nut
    r_ci_degrad = bgc.r_ci_degrad
    return (
          Ci_phy_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_PHY, thr_ci_food_het)
        + Ci_free_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_free, thr_ci_food_het)
        + Ci_pom_het(r_ci_food_het, Uz, r_pom_het, Kpom, POM, HET, Ci_POM, thr_ci_food_het)
        - Ci_het_pom(r_het_pom, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
            Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, O₂, O2_suboxic, thr_ci_food_het) 
        - Ci_het_dom(r_het_nut, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
             Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, thr_ci_food_het)            
        - Ci_het_degrad(r_ci_degrad, Ci_HET)          
    )
end
@inline function (bgc::OXYDEP)(::Val{:Ci_POM},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)
    r_ci_degrad = bgc.r_ci_degrad
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    r_ci_food_het = bgc.r_ci_food_het
    thr_ci_food_het = bgc.thr_ci_food_het
    r_het_pom = bgc.r_het_pom
    r_phy_het = bgc.r_phy_het
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_pom_dom = bgc.r_pom_dom
    r_pom_nut_nut = bgc.r_pom_nut_nut
    r_phy_pom = bgc.r_phy_pom
    O2_suboxic = bgc.O2_suboxic
    Kphy = bgc.Kphy
    Kpom = bgc.Kpom
    return (
          Ci_het_pom(r_het_pom, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
             Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, O₂, O2_suboxic, thr_ci_food_het) 
        - Ci_pom_het(r_ci_food_het, Uz, r_pom_het, Kpom, POM, HET, Ci_POM, thr_ci_food_het)
        - Ci_pom_dom( r_pom_nut_oxy, r_pom_nut_nut, r_pom_dom, O2_suboxic, 
             NUT, POM, O₂, Ci_POM)      
        + Ci_phy_pom(r_phy_pom, P, Ci_PHY)             
        - Ci_pom_degrad(r_ci_degrad, Ci_POM)    
    )
end
@inline function (bgc::OXYDEP)(::Val{:Ci_DOM},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)
    r_ci_degrad = bgc.r_ci_degrad
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    r_ci_food_het = bgc.r_ci_food_het
    thr_ci_food_het = bgc.thr_ci_food_het
    r_het_nut = bgc.r_het_nut
    r_phy_het = bgc.r_phy_het
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_pom_dom = bgc.r_pom_dom
    r_pom_nut_nut = bgc.r_pom_nut_nut
    r_phy_dom = bgc.r_phy_dom
    O2_suboxic = bgc.O2_suboxic
    Kphy = bgc.Kphy
    Kpom = bgc.Kpom
    return (
         Ci_het_dom(r_het_nut, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
             Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, thr_ci_food_het)   
        + Ci_pom_dom( r_pom_nut_oxy, r_pom_nut_nut, r_pom_dom, O2_suboxic, 
             NUT, POM, O₂, Ci_POM)     
        + Ci_phy_dom(r_phy_dom, P, Ci_PHY)                                    
        - Ci_dom_degrad(r_ci_degrad, Ci_DOM)          
    )
end

end # if Ci_

############################################################################################
# Coefficients from Garcia and Gordon (1992)
const A1 = -173.4292
const A2 = 249.6339
const A3 = 143.3483
const A4 = -21.8492
const A5 = -0.033096
const A6 = 0.014259
const B1 = -0.035274
const B2 = 0.001429
const B3 = -0.00007292
const C1 = 0.0000826

""" Function to calculate oxygen saturation in seawater """
function oxygen_saturation(T::Float64, S::Float64, P::Float64)::Float64

    T_kelvin = T + 273.15  # Convert temperature to Kelvin

    # Calculate the natural logarithm of oxygen saturation concentration
    ln_O2_sat =
        A1 +
        A2 * (100 / T_kelvin) +
        A3 * log(T_kelvin / 100) +
        A4 * T_kelvin / 100 +
        A5 * (T_kelvin / 100)^2 +
        A6 * (T_kelvin / 100)^3 +
        S * (B1 + B2 * (T_kelvin / 100) + B3 * (T_kelvin / 100)^2) +
        C1 * S^2

    # Oxygen saturation concentration in µmol/kg
    O2_sat = exp(ln_O2_sat) * 44.66

    # Pressure correction factor (Weiss, 1970) for pressure in atm
    P_corr = 1.0 + P * (5.6e-6 + 2.0e-11 * P)

    # Adjusted oxygen saturation with pressure correction
    return (O2_sat * P_corr)
end

""" Sc, Schmidt number for O2  following Wanninkhof 2014 """
@inline function OxygenSchmidtNumber(T::Float64)::Float64
    return ((1920.4 - 135.6 * T + 5.2122 * T^2 - 0.10939 * T^3 + 0.00093777 * T^4))
    # can be replaced by PolynomialParameterisation{4}((a, b, c, d, e)) i.e.:
    #    a = 1953.4, b = - 128.0, c = 3.9918, d = -0.050091, e = 0.00093777  
    # Sc = PolynomialParameterisation{4}((a, b, c, d, e))
end

""" WindDependence, [mmol m-2s-1], Oxygen Sea Water Flux """
function WindDependence(windspeed::Float64)::Float64
    return (0.251 * windspeed^2.0) #ko2o=0.251*windspeed^2*(Sc/660)^(-0.5)  Wanninkhof 2014
end

""" OxygenSeaWaterFlux, [mmol m-2s-1], Oxygen Sea Water Flux """
function OxygenSeaWaterFlux(T::Float64, S::Float64, P::Float64, O₂::Float64, windspeed::Float64)::Float64
    return (
        WindDependence(windspeed) * (OxygenSchmidtNumber(T) / 660.0)^(-0.5) * (O₂ - oxygen_saturation(T, S, P)) * 0.24 /
        86400.0        # 0.24 is to convert from [cm/h] to [m/day]  * 0.24  / 86400.0
    )
end

@inline nitrogen_flux(i, j, k, grid, advection, bgc::OXYDEP, tracers) =
    sinking_flux(i, j, k, grid, advection, Val(:POM), bgc, tracers) +
    sinking_flux(i, j, k, grid, advection, Val(:P), bgc, tracers)
@inline conserved_tracers(::OXYDEP) = (:NUT, :P, :HET, :POM, :DOM, :O₂)
@inline sinking_tracers(bgc::OXYDEP) = keys(bgc.sinking_velocities)

""" OXYDEP constants for low boundary conditions and switches """
const O2_suboxic = 20.0  # OXY threshold for oxic/suboxic switch (mmol/m3)
const Trel = 86400. # Relaxation time for exchange 
# with the sediments taking into account conversion from days to seconds (1/m)
# positive for flux from water to the sediments:
const b_O2_ox =       5.0 # flux of OXY at SWI, (mmol/m2/d) 
const b_O2_subox =   10.0 # flux of OXY at SWI in subox, (mmol/m2/d) 
const b_NUT_ox =     -1.0 # flux of NUT at SWI, (mmol/m2/d)
const b_NUT_subox =   7.0 # flux of NUT at SWI in subox, (mmol/m2/d) 
const b_DOM_ox =     -4.0 # flux of DOM at SWI, (mmol/m2/d) 
const b_DOM_subox = -12.0 # flux of DOM at SWI in subox, (mmol/m2/d)   
const bu = 0.001  #0.1 # Burial coeficient for lower boundary 
# (0<Bu<1), 1 - for burying (removal from the water column), (nd)
const windspeed = 5.0    # wind speed 10 m, (m/s)

# --- Sediment forcing functions (applied at the actual seafloor) ---
# These replace the old bottom BCs which only worked at k=1 (domain bottom).

@inline function _is_seafloor(i, j, k, grid)
    return bottommost_active_node(i, j, k, grid, Center(), Center(), Center())
end

@inline function _oxy_sediment(i, j, k, grid, clock, fields)
    bottom = _is_seafloor(i, j, k, grid)
    O₂ = @inbounds fields.O₂[i, j, k]
    flux = O₂ ≤ 0 ? zero(O₂) : -(F_ox(O₂, O2_suboxic) * b_O2_ox +
             F_subox(O₂, O2_suboxic) * b_O2_subox) / Trel
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(O₂))
end

@inline function _nut_sediment(i, j, k, grid, clock, fields)
    bottom = _is_seafloor(i, j, k, grid)
    O₂ = @inbounds fields.O₂[i, j, k]
    NUT = @inbounds fields.NUT[i, j, k]
    flux = NUT ≤ 0 ? zero(NUT) : -(F_ox(O₂, O2_suboxic) * b_NUT_ox +
             F_subox(O₂, O2_suboxic) * b_NUT_subox) / Trel
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(O₂))
end

@inline function _dom_sediment(i, j, k, grid, clock, fields)
    bottom = _is_seafloor(i, j, k, grid)
    O₂ = @inbounds fields.O₂[i, j, k]
    flux = -(F_ox(O₂, O2_suboxic) * b_DOM_ox +
             F_subox(O₂, O2_suboxic) * b_DOM_subox) / Trel
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(O₂))
end

@inline function _P_burial(i, j, k, grid, clock, fields, w_P)
    bottom = _is_seafloor(i, j, k, grid)
    P = @inbounds fields.P[i, j, k]
    w = @inbounds w_P[i, j, k]
    flux = -bu * w * P
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(P))
end

@inline function _HET_burial(i, j, k, grid, clock, fields, w_HET)
    bottom = _is_seafloor(i, j, k, grid)
    HET = @inbounds fields.HET[i, j, k]
    w = @inbounds w_HET[i, j, k]
    flux = -bu * w * HET
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(HET))
end

@inline function _POM_burial(i, j, k, grid, clock, fields, w_POM)
    bottom = _is_seafloor(i, j, k, grid)
    POM = @inbounds fields.POM[i, j, k]
    w = @inbounds w_POM[i, j, k]
    flux = -bu * w * POM
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(POM))
end

if Ci_
@inline function _Ci_PHY_burial(i, j, k, grid, clock, fields, w_Ci_PHY)
    bottom = _is_seafloor(i, j, k, grid)
    val = @inbounds fields.Ci_PHY[i, j, k]
    w = @inbounds w_Ci_PHY[i, j, k]
    flux = -bu * w * val
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(val))
end

@inline function _Ci_HET_burial(i, j, k, grid, clock, fields, w_Ci_HET)
    bottom = _is_seafloor(i, j, k, grid)
    val = @inbounds fields.Ci_HET[i, j, k]
    w = @inbounds w_Ci_HET[i, j, k]
    flux = -bu * w * val
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(val))
end

@inline function _Ci_POM_burial(i, j, k, grid, clock, fields, w_Ci_POM)
    bottom = _is_seafloor(i, j, k, grid)
    val = @inbounds fields.Ci_POM[i, j, k]
    w = @inbounds w_Ci_POM[i, j, k]
    flux = -bu * w * val
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(val))
end
end # if Ci_

"""
    oxydep_sediment_forcings(biogeochemistry)

Return a NamedTuple of DiscreteForcing objects that apply sediment-water interface
fluxes at the actual seafloor (bottommost active cell). Use this instead of the
bottom BCs in `bgh_oxydep_boundary_conditions` which only work when the seafloor
coincides with k=1 of the domain.
"""
function oxydep_sediment_forcings(biogeochemistry)
    bgc = biogeochemistry.underlying_biogeochemistry
    w_P   = biogeochemical_drift_velocity(bgc, Val(:P)).w
    w_HET = biogeochemical_drift_velocity(bgc, Val(:HET)).w
    w_POM = biogeochemical_drift_velocity(bgc, Val(:POM)).w

    base = (
        O₂  = DiscreteForcing(_oxy_sediment),
        NUT = DiscreteForcing(_nut_sediment),
        DOM = DiscreteForcing(_dom_sediment),
        P   = DiscreteForcing(_P_burial;   parameters=w_P),
        HET = DiscreteForcing(_HET_burial; parameters=w_HET),
        POM = DiscreteForcing(_POM_burial; parameters=w_POM),
    )

    if Ci_
        w_Ci_PHY = biogeochemical_drift_velocity(bgc, Val(:Ci_PHY)).w
        w_Ci_HET = biogeochemical_drift_velocity(bgc, Val(:Ci_HET)).w
        w_Ci_POM = biogeochemical_drift_velocity(bgc, Val(:Ci_POM)).w
        return merge(base, (
            Ci_PHY = DiscreteForcing(_Ci_PHY_burial; parameters=w_Ci_PHY),
            Ci_HET = DiscreteForcing(_Ci_HET_burial; parameters=w_Ci_HET),
            Ci_POM = DiscreteForcing(_Ci_POM_burial; parameters=w_Ci_POM),
        ))
    else
        return base
    end
end

""" BGC boundary conditions (O₂ surface gas exchange only) """
function bgh_oxydep_boundary_conditions(biogeochemistry, Nz)

    Oxy_top_cond(i, j, grid, clock, fields) = @inbounds (OxygenSeaWaterFlux(
        fields.T[i, j, Nz],
        fields.S[i, j, Nz],
        0.0,                # sea surface pressure
        fields.O₂[i, j, Nz],
        windspeed,
    ))

    OXY_top = FluxBoundaryCondition(Oxy_top_cond; discrete_form = true)
    oxy_bcs = FieldBoundaryConditions(top = OXY_top)

    return (O₂ = oxy_bcs,)
end

end  # module