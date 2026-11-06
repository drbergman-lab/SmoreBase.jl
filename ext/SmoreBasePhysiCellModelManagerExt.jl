module SmoreBasePhysiCellModelManagerExt

using SmoreBase
using PhysiCellModelManager
using PhysiCellModelManager: AbstractTrial, SimulationPopulationTimeSeries

const PCMM = PhysiCellModelManager

# ── entry points ───────────────────────────────────────────────────────────────

function SmoreBase.cmDataFromPCMM(T::AbstractTrial; kwargs...)
    return _fromMonadIDs(unique(monadIDs(T)); kwargs...)
end

SmoreBase.cmDataFromPCMM(output::PCMMOutput; kwargs...) = SmoreBase.cmDataFromPCMM(output.trial; kwargs...)

SmoreBase.cmDataFromPCMM(monads::AbstractVector{<:Monad}; kwargs...) =
    _fromMonadIDs(unique([m.id for m in monads]); kwargs...)

function SmoreBase.cmDataFromPCMM(monad::Monad; kwargs...)
    sim_ids = simulationIDs(monad)
    data = _build([sim_ids], ["monad$(monad.id)"]; kwargs...)
    return (data=data, cm_sample=nothing, monad_ids=[monad.id], simulation_ids=[sim_ids])
end

function SmoreBase.cmDataFromPCMM(simulations::AbstractVector{<:Simulation}; kwargs...)
    sim_ids = [s.id for s in simulations]
    data = _build([sim_ids], ["ps1"]; kwargs...)
    return (data=data, cm_sample=nothing, monad_ids=Int[], simulation_ids=[sim_ids])
end

function SmoreBase.cmDataFromPCMM(simulation::Simulation; kwargs...)
    throw(ArgumentError(
        "cmDataFromPCMM got a single Simulation ($(simulation.id)); one replicate has no " *
        "standard deviation. Pass its Monad (`Monad(simulation)`) or a vector of replicate Simulations."))
end

# ── helpers ────────────────────────────────────────────────────────────────────

function _fromMonadIDs(monad_ids::AbstractVector{<:Integer}; kwargs...)
    isempty(monad_ids) && throw(ArgumentError("cmDataFromPCMM: no monads found"))
    if length(monad_ids) == 1
        return SmoreBase.cmDataFromPCMM(Monad(only(monad_ids)); kwargs...)
    end

    # One row per monad, sorted by the varied parameters; constant columns removed.
    df = monadsTable(monad_ids)
    ordered_ids = Int.(df[!, :MonadID])
    id_cols = Set(String.(shortLocationVariationID.(String, projectLocations().varied)))
    varied = [(name, df[!, name]) for name in names(df) if name != "MonadID" && !(name in id_cols)]

    labels = SmoreBase._variedLabels(varied, length(ordered_ids))
    isnothing(labels) && (labels = ["monad$id" for id in ordered_ids])
    allunique(labels) || (labels = ["monad$(id): $(l)" for (id, l) in zip(ordered_ids, labels)])

    numeric = [(name, vals) for (name, vals) in varied if all(v -> v isa Real, vals)]
    cm_sample = isempty(numeric) ? nothing :
        CMSample(reduce(hcat, [Float64.(vals) for (_, vals) in numeric]);
                 names=[String(name) for (name, _) in numeric])

    sim_ids = [simulationIDs(Monad(id)) for id in ordered_ids]
    data = _build(sim_ids, labels; kwargs...)
    return (data=data, cm_sample=cm_sample, monad_ids=ordered_ids, simulation_ids=sim_ids)
end

function _build(sim_ids::AbstractVector{<:AbstractVector{<:Integer}}, labels::AbstractVector{<:AbstractString};
                cell_types=:all, include_dead::Bool=false, kwargs...)
    series = [[_readSeries(id, include_dead) for id in ids] for ids in sim_ids]
    variables = _resolveCellTypes(cell_types, series[1][1])
    groups = [[(id=id, time=s.time, counts=_counts(s, variables, id)) for (id, s) in zip(ids, ss)]
              for (ids, ss) in zip(sim_ids, series)]
    return SmoreBase._cmDataFromReplicates(groups; variable_labels=variables,
                                           cm_param_set_labels=labels, kwargs...)
end

function _readSeries(id::Integer, include_dead::Bool)
    spts = SimulationPopulationTimeSeries(id; include_dead=include_dead, verbose=false)
    ismissing(spts) && throw(ArgumentError(
        "simulation $id has no output on disk (deleted, pruned, or not yet run)"))
    return (time=Float64.(spts.time), cell_count=spts.cell_count)
end

_resolveCellTypes(c::AbstractString, _) = [String(c)]
_resolveCellTypes(c::AbstractVector{<:AbstractString}, _) =
    isempty(c) ? throw(ArgumentError("cell_types is empty")) : String.(c)
function _resolveCellTypes(c::Symbol, s)
    c === :all || throw(ArgumentError("cell_types must be :all, a name, or a vector of names; got :$c"))
    return sort!(collect(String, keys(s.cell_count)))
end

function _counts(s, variables, id)
    M = Matrix{Float64}(undef, length(s.time), length(variables))
    for (j, name) in enumerate(variables)
        haskey(s.cell_count, name) || throw(ArgumentError(
            "cell type \"$name\" not found in simulation $id; available: " *
            join(sort!(collect(keys(s.cell_count))), ", ")))
        M[:, j] .= s.cell_count[name]
    end
    return M
end

end
