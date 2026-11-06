module SmoreBasePhysiCellModelManagerExt

using SmoreBase
using Statistics
using PhysiCellModelManager
using PhysiCellModelManager: AbstractTrial, AbstractMonad, SimulationPopulationTimeSeries

# ── CMData ─────────────────────────────────────────────────────────────────────

"""
    CMData(T; cell_types=:all, include_dead=false, time_unit=:min, obs_times=nothing, time_tol=1e-6)

Build a `CMData` from PhysiCellModelManager.jl (PCMM) output: one CM parameter set per PCMM
`Monad`, holding the replicate mean (`μ`) and sample standard deviation (`σ`) of the population
count of each cell type. Available after `using PhysiCellModelManager`.

# Arguments
`T` may be
- a `Sampling` or `Trial`, or the `PCMMOutput` that `run` returns for one: one CM parameter set
  per monad it contains;
- a vector of `Monad`s: one CM parameter set per monad;
- a `Monad`: one CM parameter set from its replicates;
- a vector of `Simulation`s from one monad: one CM parameter set from those replicates only
  (an error if they come from different monads);
- a `Simulation`: one CM parameter set from that single run.

# Keyword arguments
- `cell_types` — `:all` (default: every cell type in the model, sorted by name), one name such
  as `"default"`, or a vector of names. Each becomes one CMData variable, in that order.
- `include_dead::Bool=false` — also count dead cells.
- `time_unit=:min` — unit of the returned `times` and of `obs_times`. PhysiCell saves in
  minutes; pass `:h` or `:d` (also `:s`, `:w`, `:mo`, `:y`; the same symbols PCMM's plots take).
- `obs_times` — `nothing` (default: every save time) or the times, in `time_unit`, to keep.
  Each must match a save time to within `time_tol` (absolute, in `time_unit`); a requested time
  that was not saved is an error.

# Returns
A `CMData` of shape `[n_times × n_cell_types × 1 × n_cm_param_sets]`. Parameter sets are ordered
as `monadsTable(T)` sorts the monads (by every varied parameter, the first as primary key), and
labelled `"name=value, …"` from the varied parameters, or `"monad<id>"` when none vary.
[`CMSample(T)`](@ref SmoreBase.CMSample(::PhysiCellModelManager.AbstractTrial)) returns the
matching parameter values, one row per parameter set in the same order.

All replicates of a monad must share save times, and so must all parameter sets (after
`obs_times` selection), and every simulation must have the same cell types.

`σ` is not checked here: it is 0 where every replicate agrees (typically t = 0) and `NaN` for a
single replicate. `SMFitProblem` rejects such data when the loss is `GaussianNLL`; drop those
times with `obs_times`, or use a `CustomLoss`.

PCMM computes each simulation's population counts by parsing every output snapshot the first
time they are requested, then saves them in the simulation's `summary/` folder; later calls on
the same simulations read that file instead.

# Example
```julia
using SmoreBase, PhysiCellModelManager
initializeModelManager("path/to/project")

sampling = Sampling(1)
data = CMData(sampling; cell_types="default", time_unit=:h, obs_times=[24.0, 48.0, 72.0])
cm_sample = CMSample(sampling)   # row k = varied parameter values of data's parameter set k
```
"""
SmoreBase.CMData(T::AbstractTrial; kwargs...) = _cmData(_parameterSets(T)...; kwargs...)

SmoreBase.CMData(output::PCMMOutput; kwargs...) = SmoreBase.CMData(output.trial; kwargs...)

SmoreBase.CMData(monads::AbstractVector{<:Monad}; kwargs...) =
    _cmData(_parameterSets(unique([m.id for m in monads]))...; kwargs...)

function SmoreBase.CMData(simulations::AbstractVector{<:Simulation}; kwargs...)
    isempty(simulations) && throw(ArgumentError("CMData: the vector of simulations is empty"))
    sim_ids = [s.id for s in simulations]
    monad_ids = unique(reduce(vcat, [monadIDs(s) for s in simulations]))
    length(monad_ids) == 1 || throw(ArgumentError(
        "CMData: a vector of simulations is one CM parameter set, so they must come from one " *
        "monad; simulations $(sim_ids) belong to monads $(monad_ids). Pass the monads instead."))
    return _cmData([sim_ids], ["monad$(only(monad_ids))"]; kwargs...)
end

# ── CMSample ───────────────────────────────────────────────────────────────────

"""
    CMSample(T) -> AbstractCMSample

The numeric parameters that vary across the monads of `T` (a `Sampling`, `Trial`, `PCMMOutput`,
or vector of `Monad`s), as a `[n_monads × n_varied]` `CMSample`: row `k` holds the values of
parameter set `k` of `CMData(T)`, i.e. it lines up with the 4th dimension of the data arrays.
Columns are named by PCMM's short parameter names; a regular grid (e.g. from a `GridVariation`)
comes back as a `GridCMSample`. Non-numeric varied columns (e.g. differing config folders) are
left out. Available after `using PhysiCellModelManager`.

Throws an `ArgumentError` if no numeric parameter varies (e.g. `T` is a single monad).

# Example
```julia
cm_sample = CMSample(Sampling(1))
cm_sample.names    # varied parameter names
cm_sample.params   # [n_monads × n_varied]
```
"""
function SmoreBase.CMSample(T::AbstractTrial)
    T isa AbstractMonad && throw(ArgumentError(
        "CMSample: a single $(nameof(typeof(T))) has no varied parameters; pass a Sampling, Trial or vector of Monads"))
    return _cmSample(unique(monadIDs(T)))
end

SmoreBase.CMSample(output::PCMMOutput) = SmoreBase.CMSample(output.trial)
SmoreBase.CMSample(monads::AbstractVector{<:Monad}) = _cmSample(unique([m.id for m in monads]))

function _cmSample(monad_ids)
    _, _, varied = _variedTable(monad_ids)
    numeric = [(name, vals) for (name, vals) in varied if all(v -> v isa Real, vals)]
    isempty(numeric) && throw(ArgumentError("CMSample: no numeric parameter varies across monads $(monad_ids)"))
    return CMSample(reduce(hcat, [Float64.(vals) for (_, vals) in numeric]);
                    names=[String(name) for (name, _) in numeric])
end

# ── parameter sets ─────────────────────────────────────────────────────────────

# (simulation ids per parameter set, labels) for a PCMM object.
function _parameterSets(T::AbstractTrial)
    T isa Simulation && return [[T.id]], ["monad$(only(monadIDs(T)))"]
    T isa Monad && return [simulationIDs(T)], ["monad$(T.id)"]
    return _parameterSets(unique(monadIDs(T)))
end

function _parameterSets(monad_ids::AbstractVector{<:Integer})
    isempty(monad_ids) && throw(ArgumentError("CMData: no monads found"))
    length(monad_ids) == 1 && return _parameterSets(Monad(only(monad_ids)))
    ordered_ids, _, varied = _variedTable(monad_ids)
    labels = isempty(varied) ? ["monad$id" for id in ordered_ids] :
        [join(["$(name)=$(vals[k])" for (name, vals) in varied], ", ") for k in eachindex(ordered_ids)]
    allunique(labels) || (labels = ["monad$(id): $(l)" for (id, l) in zip(ordered_ids, labels)])
    return [simulationIDs(Monad(id)) for id in ordered_ids], labels
end

# One row per monad, sorted by the varied parameters, constant and variation-ID columns removed.
# Returns (monad ids in row order, the table, [(column name, values)] for the varied parameters).
function _variedTable(monad_ids)
    df = monadsTable(monad_ids)
    id_cols = Set(String.(shortLocationVariationID.(String, projectLocations().varied)))
    varied = [(name, df[!, name]) for name in names(df) if name != "MonadID" && !(name in id_cols)]
    return Int.(df[!, :MonadID]), df, varied
end

# ── reading and aggregation ────────────────────────────────────────────────────

function _cmData(sim_ids::AbstractVector{<:AbstractVector{<:Integer}}, labels; include_dead::Bool=false, kwargs...)
    groups = [[id => _readSeries(id, include_dead) for id in ids] for ids in sim_ids]
    return _cmDataFromSeries(groups, labels; kwargs...)
end

function _readSeries(id::Integer, include_dead::Bool)
    spts = SimulationPopulationTimeSeries(id; include_dead=include_dead, verbose=false)
    ismissing(spts) && throw(ArgumentError(
        "simulation $id has no output on disk (deleted, pruned, or not yet run)"))
    return spts
end

# Aggregate population time series into a `CMData`. `groups[k]` holds the replicates of parameter
# set `k` as `simulation_id => series` pairs; a series needs `.time` (PhysiCell minutes) and
# `.cell_count` (cell type name → counts), as a PCMM `SimulationPopulationTimeSeries` has.
# Kept free of database calls so it is tested without a PhysiCell project.
function _cmDataFromSeries(groups::AbstractVector, labels::AbstractVector{<:AbstractString};
                           cell_types=:all, time_unit=:min,
                           obs_times::Union{Nothing,AbstractVector{<:Real}}=nothing,
                           time_tol::Real=1e-6)
    isempty(groups) && throw(ArgumentError("CMData: no CM parameter sets"))
    length(labels) == length(groups) ||
        throw(ArgumentError("got $(length(labels)) labels for $(length(groups)) CM parameter sets"))
    for (g, label) in zip(groups, labels)
        isempty(g) && throw(ArgumentError("CM parameter set $label has no simulations"))
    end
    isnothing(obs_times) || isempty(obs_times) && throw(ArgumentError("obs_times is empty"))

    first_id, first_series = groups[1][1]
    roster = Set(keys(first_series.cell_count))
    for g in groups, (id, s) in g
        Set(keys(s.cell_count)) == roster || throw(ArgumentError(
            "simulations $first_id and $id have different cell types " *
            "($(join(sort!(collect(roster)), ", ")) vs $(join(sort!(collect(keys(s.cell_count))), ", ")))"))
    end
    variables = _resolveCellTypes(cell_types, roster)

    stats = map(zip(groups, labels)) do (g, label)
        t = g[1][2].time
        for (id, s) in g
            s.time == t || throw(ArgumentError(
                "simulations $(g[1][1]) and $id of CM parameter set $label have different save times"))
        end
        t_u = _convertTime(Float64.(t), time_unit)
        rows = isnothing(obs_times) ? collect(eachindex(t_u)) :
               [_matchTime(t_u, τ, time_tol, label) for τ in obs_times]
        Y = Array{Float64}(undef, length(rows), length(variables), length(g))
        for (r, (_, s)) in enumerate(g), (j, name) in enumerate(variables)
            Y[:, j, r] .= s.cell_count[name][rows]
        end
        μ = dropdims(mean(Y; dims=3); dims=3)
        σ = size(Y, 3) == 1 ? fill(NaN, size(μ)) : dropdims(std(Y; dims=3); dims=3)
        (t_u[rows], μ, σ)
    end

    t = stats[1][1]
    for (k, s) in enumerate(stats)
        length(s[1]) == length(t) && all(isapprox.(s[1], t; atol=time_tol)) || throw(ArgumentError(
            "CM parameter sets $(labels[1]) and $(labels[k]) do not share save times; pass " *
            "`obs_times` to select common times"))
    end

    return CMData(μ=cat((s[2] for s in stats)...; dims=3), σ=cat((s[3] for s in stats)...; dims=3),
                  times=t, variables=variables, cm_param_sets=collect(String, labels),
                  dim_order=[:times, :variables, :cm_param_sets])
end

_resolveCellTypes(c::AbstractString, roster) = _resolveCellTypes([c], roster)
function _resolveCellTypes(c::AbstractVector{<:AbstractString}, roster)
    isempty(c) && throw(ArgumentError("cell_types is empty"))
    for name in c
        name in roster || throw(ArgumentError(
            "cell type \"$name\" not found; available: $(join(sort!(collect(roster)), ", "))"))
    end
    return String.(c)
end
function _resolveCellTypes(c::Symbol, roster)
    c === :all || throw(ArgumentError("cell_types must be :all, a name, or a vector of names; got :$c"))
    return sort!(collect(String, roster))
end

# Minutes per unit, matching the symbols PCMM's plotting recipes accept for `time_unit`.
const _MINUTES_PER = Dict{Symbol,Float64}()
for (syms, m) in (((:min, :minute, :minutes), 1.0), ((:s, :second, :seconds), 1 / 60),
                  ((:h, :hour, :hours), 60.0), ((:d, :day, :days), 1440.0),
                  ((:w, :week, :weeks), 10080.0),
                  ((:mo, :month, :months), 43200.0),   # 30-day month, as in PCMM
                  ((:y, :year, :years), 525600.0))     # 365-day year, as in PCMM
    for sym in syms
        _MINUTES_PER[sym] = m
    end
end

function _convertTime(t_min::AbstractVector{<:Real}, unit)
    haskey(_MINUTES_PER, Symbol(unit)) || throw(ArgumentError(
        "invalid time_unit :$unit; use one of :min, :s, :h, :d, :w, :mo, :y"))
    return t_min ./ _MINUTES_PER[Symbol(unit)]
end

function _matchTime(t::AbstractVector{<:Real}, τ::Real, tol::Real, label)
    r = findfirst(x -> isapprox(x, τ; atol=tol), t)
    isnothing(r) && throw(ArgumentError(
        "requested time $τ is not among the save times of CM parameter set $label " *
        "(saved: $(isempty(t) ? "none" : "$(first(t)) … $(last(t)), $(length(t)) times"); " *
        "matched within time_tol=$tol)"))
    return r
end

end
