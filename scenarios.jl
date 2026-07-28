module ScenarioConfigs

using JSON3

export ScenarioConfig, SCENARIOS, get_scenario, resolve_paths

Base.@kwdef struct ScenarioConfig
    grid_file::String
    forcing_file::String
    atmospheric_forcing_dir::String
    hotstart_file::String = "snapshots_ocean_1.nc"
end

function load_scenarios(path::AbstractString)
    raw = JSON3.read(read(path, String))
    return Dict(
        String(name) => ScenarioConfig(;
            grid_file = String(obj.grid_file),
            forcing_file = String(obj.forcing_file),
            atmospheric_forcing_dir = String(obj.atmospheric_forcing_dir),
            hotstart_file = haskey(obj, :hotstart_file) ? String(obj.hotstart_file) : "snapshots_ocean.nc",
        )
        for (name, obj) in pairs(raw)
    )
end

function get_scenario(name::AbstractString)
    if !haskey(SCENARIOS, name)
        available = join(sort(collect(keys(SCENARIOS))), ", ")
        error("Unknown scenario '$(name)'. Available scenarios: $(available)")
    end
    return SCENARIOS[name]
end

function resolve_paths(scenario_name::AbstractString, config::ScenarioConfig, project_root::AbstractString)
    input_dir = joinpath(project_root, "data", "input")
    output_dir = joinpath(project_root, "data", "output", scenario_name)
    return (
        grid_path = joinpath(input_dir, config.grid_file),
        forcing_path = joinpath(input_dir, config.forcing_file),
        atmospheric_forcing_path = joinpath(input_dir, config.atmospheric_forcing_dir),
        results_path = output_dir,
        hotstart_path = joinpath(output_dir, config.hotstart_file),
    )
end

const SCENARIOS = load_scenarios(joinpath(@__DIR__, "scenarios.json"))

end
