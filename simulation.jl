#!/usr/bin/env julia
using Oceananigans
using Oceananigans.Units
using ClimaOcean
using SeawaterPolynomials.TEOS10
using FjordSim
using FjordSim.FDatasets
using ArgParse
using Dates: DateTime

#if !@isdefined(Ci_)
    const Ci_ = false  # true: include Ci fields and calculations; false: skip them
#else
#    const Ci_ = true
#
#end

include("Oxydep.jl")
include("scenarios.jl")
using .OXYDEPModel
using .ScenarioConfigs

const _uniform_IC = true # if false, read initial conditions from "snapshots_ocean_XX.nc"
import Oceananigans.Advection: cell_advection_timescale
using NumericalEarth.EarthSystemModels: EarthSystemModel
cell_advection_timescale(model::EarthSystemModel) =
    cell_advection_timescale(model.ocean.model)

const FT = Oceananigans.defaults.FloatType

function select_arch(arch_str::AbstractString)
    arch = lowercase(strip(arch_str))

    if arch == "cpu"
        return CPU()
    elseif arch == "gpu"
        return GPU()  # throws if no CUDA GPU
    else
        error("Invalid --arch value: $(repr(arch_str)). Use 'cpu' or 'gpu'.")
    end
end


# ----------------------------------------------------------
# Command-line argument parser
# ----------------------------------------------------------
function default_paths()
    base_dir = @__DIR__

    # Docker / cloud: the Dockerfile sets SIMULATION_LAUNCHER and entrypoint.sh
    # passes explicit --paths anyway.  Local dev: no such env var.
    grid     = joinpath(base_dir, "data", "input", "bathymetry_25to70.nc")
    forcing  = joinpath(base_dir, "data", "input", "forcing_25to70.nc") #"forcing_drammen_w9Bo.nc")
    atm      = joinpath(base_dir, "data", "input", "JRA55")
    results  = joinpath(base_dir, "data", "output", "inner_oslofjord")

    return (; grid, forcing, atm, results)
end

function parse_commandline()
    s = ArgParseSettings()
    paths = default_paths()

    @add_arg_table! s begin
        "--grid_path"
        help = "Path to the bathymetry NetCDF file."
        arg_type = String
        default = paths.grid

        "--forcing_path"
        help = "Path to the forcing NetCDF file."
        arg_type = String
        default = paths.forcing

        "--atmospheric_forcing_path"
        help = "Path to the atmospheric forcing directory."
        arg_type = String
        default = paths.atm

        "--results_path"
        help = "Directory where results are stχored."
        arg_type = String
        default = paths.results

        "--arch"
        help = "Compute architecture: cpu or gpu."
        arg_type = String
        default = "gpu"

        "--scenario"
        help = "Scenario name from scenarios.json. If provided, overrides individual path arguments."
        arg_type = String
        default = ""

        "--stop_days"
        help = "Simulation duration in days. If omitted, the hardcoded default is used."
        arg_type = Int
        default = 0

    end
    return parse_args(s)
end

# ----------------------------------------------------------
# Main simulation setup
# ----------------------------------------------------------
function main()
    args = parse_commandline()
    arch = select_arch(args["arch"])

    # If a scenario is provided, resolve its paths and override the individual
    # path arguments. Otherwise, use the explicit per-path arguments / defaults.
    if args["scenario"] !== nothing && args["scenario"] != ""
        project_root = get(ENV, "PROJECT_ROOT", @__DIR__)
        scenario_name = args["scenario"]
        scenario = ScenarioConfigs.get_scenario(scenario_name)
        scenario_paths = ScenarioConfigs.resolve_paths(scenario_name, scenario, project_root)
        grid_path = scenario_paths.grid_path
        forcing_path = scenario_paths.forcing_path
        atmospheric_forcing_path = scenario_paths.atmospheric_forcing_path
        results_path = scenario_paths.results_path
        println("Running simulation with scenario: $(scenario_name)")
    else
        grid_path = args["grid_path"]
        forcing_path = args["forcing_path"]
        atmospheric_forcing_path = args["atmospheric_forcing_path"]
        results_path = args["results_path"]
    end

    println("Running simulation with:")
    println("  arch = $(args["arch"])")
    println("  selected_architecture = $(typeof(arch))")
    println("  grid_path = $(grid_path)")
    println("  forcing_path = $(forcing_path)")
    println("  atmospheric_forcing_path = $(atmospheric_forcing_path)")
    println("  results_path = $(results_path)")

    grid = ImmersedBoundaryGrid(grid_path, arch, (7, 7, 7))
    buoyancy = SeawaterBuoyancy(FT, equation_of_state=TEOS10EquationOfState(FT))
    closure = (
        CATKEVerticalDiffusivity(minimum_tke = 5.5e-6), #% (minimum_tke = 7e-6)
        # Biharmonic (4th-order) diffusion for physical + BGC tracers.
        # Ci tracers get κ=0 here: biharmonic is anti-diffusive near sharp
        # gradients (river source) and creates negative oscillations.
        ##Oceananigans.TurbulenceClosures.HorizontalScalarBiharmonicDiffusivity(ν = 15,
        ##    κ = Ci_ ?
        ##        (T=50, S=50, e=0, NUT=50, P=50, HET=50, POM=50, DOM=50, O₂=50,
        ##         Ci_free=0, Ci_PHY=0, Ci_HET=0, Ci_POM=0, Ci_DOM=0) :
        ##        (T=50, S=50, e=0, NUT=50, P=50, HET=50, POM=50, DOM=50, O₂=50),
        ##),
        # 2nd-order (Laplacian) horizontal diffusion for all tracers.
        # Unlike biharmonic, Laplacian diffusion is positive-definite: it cannot
        # create new extrema, so it smears the sharp river-source gradient without
        # generating undershoots.  κ=10 m²/s gives ~2.25 h e-folding at 300 m grid.
        Oceananigans.TurbulenceClosures.HorizontalScalarDiffusivity(
            κ = Ci_ ?
                (T=10, S=10, e=0, NUT=10, P=10, HET=10, POM=10, DOM=10, O₂=10,
                 Ci_free=10, Ci_PHY=10, Ci_HET=10, Ci_POM=10, Ci_DOM=10) :
                (T=10, S=10, e=0, NUT=10, P=10, HET=10, POM=10, DOM=10, O₂=10),
        ),
     )    
#¤     tracer_advection = ()
    # Ci tracers use UpwindBiased(order=5) instead of WENO.
    # WENO is not positive-definite and creates Gibbs-like oscillations (negative
    # undershoot) near the sharp concentration gradient at the river source.
    # UpwindBiased(order=5) has an inherent upwind bias that suppresses oscillations
    # while remaining 5th-order accurate away from gradients.
    tracer_advection = if Ci_
        (
        T=WENO(),
        S=WENO(),
            e=nothing,
            NUT=UpwindBiased(order=5),
            P=UpwindBiased(order=5),
            HET=UpwindBiased(order=5),
            POM=UpwindBiased(order=5),
            DOM=UpwindBiased(order=5),
            O₂=UpwindBiased(order=5),
            Ci_free=UpwindBiased(order=5),
            Ci_PHY=UpwindBiased(order=5),
            Ci_HET=UpwindBiased(order=5),
            Ci_POM=UpwindBiased(order=5),
            Ci_DOM=UpwindBiased(order=5),
        )
    else
        (
            T=WENO(),
            S=WENO(),
            e=nothing,
        NUT=WENO(),
        P=WENO(),
        HET=WENO(),
        POM=WENO(),
        DOM=WENO(),
        O₂=WENO(),
    )
    end
    momentum_advection = WENOVectorInvariant(FT)

    tracers = if Ci_
        (
            :T, :S, :NUT, :P, :HET, :POM, :DOM, :O₂,
            :Ci_free, :Ci_PHY, :Ci_HET, :Ci_POM, :Ci_DOM,
        )
    else
        (:T, :S, :NUT, :P, :HET, :POM, :DOM, :O₂)
    end
    
    if !_uniform_IC
        restart_paths = default_paths()
        dataset = DSResults(
            "snapshots_ocean_9.nc",
            restart_paths.results;
            start_date_time = DateTime(2025, 1, 1),
        )
        initial_conditions_base = (
            T = Metadatum(:temperature; dataset, date = last_date(dataset, :temperature)),
            S = Metadatum(:salinity; dataset, date = last_date(dataset, :salinity)),
            e = Metadatum(:e; dataset, date = last_date(dataset, :e)),
            NUT = Metadatum(:NUT; dataset, date = last_date(dataset, :NUT)),
            P = Metadatum(:P; dataset, date = last_date(dataset, :P)),
            HET = Metadatum(:HET; dataset, date = last_date(dataset, :HET)),
            POM = Metadatum(:POM; dataset, date = last_date(dataset, :POM)),
            DOM = Metadatum(:DOM; dataset, date = last_date(dataset, :DOM)),
            O₂ = Metadatum(:O₂; dataset, date = last_date(dataset, :O₂)),
            u = Metadatum(
                :u_velocity;
                dataset,
                date = last_date(dataset, :u_velocity),
            ),
            v = Metadatum(
                :v_velocity;
                dataset,
                date = last_date(dataset, :v_velocity),
            ),
        )
        if Ci_
            initial_conditions = merge(initial_conditions_base, (
                Ci_free = Metadatum(:Ci_free; dataset, date = last_date(dataset, :Ci_free)),
                Ci_PHY = Metadatum(:Ci_PHY; dataset, date = last_date(dataset, :Ci_PHY)),
                Ci_HET = Metadatum(:Ci_HET; dataset, date = last_date(dataset, :Ci_HET)),
                Ci_POM = Metadatum(:Ci_POM; dataset, date = last_date(dataset, :Ci_POM)),
                Ci_DOM = Metadatum(:Ci_DOM; dataset, date = last_date(dataset, :Ci_DOM)),
            ))
        else
            initial_conditions = initial_conditions_base
        end
    else
        initial_conditions_base = (
            T = 1.0, S = 20, NUT = 18., P = 0.01, HET = 0.01, O₂ = 150., DOM = 0.05, POM = 0.01,
        )
        if Ci_
            initial_conditions = merge(initial_conditions_base, (
                Ci_free = 0.0, Ci_PHY = 0.0, Ci_HET = 0.0, Ci_POM = 0.0, Ci_DOM = 0.0,
            ))
        else
            initial_conditions = initial_conditions_base
        end
    end
    free_surface = SplitExplicitFreeSurface(grid, cfl = 0.7)
    coriolis = HydrostaticSphericalCoriolis(FT)
    forcing_file = forcing_from_file(;
        grid=grid,
        filepath=forcing_path,
        tracers=tracers,
    )
    tbbc = top_bottom_boundary_conditions(;
        grid = grid,
        bottom_drag_coefficient = 0.003,
    )
    sobc = (v = (south = OpenBoundaryCondition(nothing),),)
    boundary_conditions = map(x -> FieldBoundaryConditions(; x...), recursive_merge(tbbc, sobc))
    biogeochemistry = OXYDEPModel.OXYDEP(grid)
    boundary_conditions = merge(boundary_conditions, OXYDEPModel.bgh_oxydep_boundary_conditions(biogeochemistry, grid.Nz))
    # Sediment forcings applied at actual seafloor (works with ImmersedBoundaryGrid)
    forcing_sed = OXYDEPModel.oxydep_sediment_forcings(biogeochemistry)
    forcing = forcing_file
    for name in keys(forcing_sed)
        if haskey(forcing, name)
            forcing = merge(forcing, NamedTuple{(name,)}(((forcing[name], forcing_sed[name]),)))
        else
            forcing = merge(forcing, NamedTuple{(name,)}((forcing_sed[name],)))
        end
    end
     atmosphere = JRA55PrescribedAtmosphere(arch, FT;
         latitude=(58.98, 59.94),
         longitude=(10.18, 11.03),
         dir=atmospheric_forcing_path,
     )
#    atmosphere = NORA3PrescribedAtmosphere(arch)
    #atmosphere = NORA3PrescribedAtmosphere(arch)
    downwelling_radiation = JRA55PrescribedRadiation(arch, FT;
        ocean_surface=SurfaceRadiationProperties(0.1, 0.96),
        latitude=(58.98, 59.94),
        longitude=(10.18, 11.03),
        dir=args["atmospheric_forcing_path"],
    )
    sea_ice = FreezingLimitedOceanTemperature()
    results_dir = results_path
    stop_days = args["stop_days"]
    stop_time = if stop_days > 0
        stop_days * days
    else
        365days
    end
    #¤biogeochemistry = nothing

    simulation = coupled_hydrostatic_simulation(
        grid,
        buoyancy,
        closure,
        tracer_advection,
        momentum_advection,
        tracers,
        initial_conditions,
        free_surface,
        coriolis,
        forcing,
        boundary_conditions,
        atmosphere,
        downwelling_radiation,
        sea_ice,
        biogeochemistry;
        results_dir,
        stop_time,
    )

    simulation.callbacks[:progress] = Callback(progress, TimeInterval(24hours))

    # Clip negative BGC tracer values to zero after every time step.
    # ScaleNegativeTracers (in OXYDEP) only runs during the BGC update; this callback
    # catches negatives produced by hydrodynamic transport (advection + diffusion).
    # T and S are intentionally excluded (legitimate negative values possible near freezing).
    # Uses clamp! on the underlying parent array so it works on both CPU and GPU.
    let _bgc_tracers_to_clip = Ci_ ?
            (:NUT, :P, :HET, :POM, :DOM, :O₂,
             :Ci_free, :Ci_PHY, :Ci_HET, :Ci_POM, :Ci_DOM) :
            (:NUT, :P, :HET, :POM, :DOM, :O₂)
        function enforce_positivity!(sim)
            m = sim.model.ocean.model
            FT = eltype(m.grid)
            for name in _bgc_tracers_to_clip
                f = m.tracers[name]
                clamp!(parent(f), zero(FT), FT(Inf))
            end
        end
        simulation.callbacks[:enforce_positivity] =
            Callback(enforce_positivity!, IterationInterval(1))
    end

    ocean_sim = simulation.model.ocean
    ocean_model = ocean_sim.model

    # χ = -0.5 → pure forward Euler: U += Δt * Gⁿ  (no use of previous tendency G⁻)
    # χ =  0.0 → standard AB2:       U += Δt * (1.5·Gⁿ - 0.5·Gⁿ⁻¹)
    # χ =  0.1 → Oceananigans default AB2 (slightly damped)
    # Applies to ALL ocean prognostic fields: u, v, w, T, S, e, and all BGC/Ci tracers.
    ocean_model.timestepper.χ = -0.5

    prefix = joinpath(results_dir, "snapshots_ocean_iof")
    output_fields_base = (
            T=ocean_model.tracers.T,
            S=ocean_model.tracers.S,
            NUT=ocean_model.tracers.NUT,
            P=ocean_model.tracers.P,
            HET=ocean_model.tracers.HET,
            POM=ocean_model.tracers.POM,
            DOM=ocean_model.tracers.DOM,
            O₂=ocean_model.tracers.O₂,
        e=ocean_model.tracers.e,
            u=ocean_model.velocities.u,
            v=ocean_model.velocities.v,
    )
    output_fields = if Ci_
        merge(output_fields_base, (
            Ci_free=ocean_model.tracers.Ci_free,
            Ci_PHY=ocean_model.tracers.Ci_PHY,
            Ci_HET=ocean_model.tracers.Ci_HET,
            Ci_POM=ocean_model.tracers.Ci_POM,
            Ci_DOM=ocean_model.tracers.Ci_DOM,
        ))
    else
        output_fields_base
    end
    ocean_sim.output_writers[:ocean] = NetCDFWriter(
        ocean_model,
        output_fields;
        filename="$prefix",
        schedule=TimeInterval(12hours),
        overwrite_existing=true,
    )

#    conjure_time_step_wizard!(simulation; cfl=0.1, max_Δt=45, max_change=1.01)
    conjure_time_step_wizard!(simulation; cfl=0.1, max_Δt=3minutes, max_change=1.01)
    run!(simulation)
end

# ----------------------------------------------------------
# Run script
# ----------------------------------------------------------
main()
