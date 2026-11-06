"""
    cmDataFromPCMM(T; cell_types=:all, include_dead=false, time_scale=1/60, obs_times=nothing,
                   time_tol=1e-6, sd_floor=nothing) -> NamedTuple

Build a `CMData` from PhysiCellModelManager.jl (PCMM) simulation output: one CM parameter set
per PCMM `Monad`, holding the replicate mean and sample standard deviation of the population
count of each cell type.

Requires the `SmoreBasePhysiCellModelManagerExt` extension, which loads with
`using PhysiCellModelManager`.

# Arguments
`T` may be
- a `Sampling` or `Trial` (or the `PCMMOutput` returned by `run` on one) — one CM parameter set
  per monad it contains;
- a vector of `Monad`s — one CM parameter set per monad;
- a single `Monad` — one CM parameter set from its replicates;
- a vector of `Simulation`s — treated as replicates of a single CM parameter set.

A lone `Simulation` is rejected: one replicate has no standard deviation.

# Keyword arguments
- `cell_types` — `:all` (default: every cell type in the model, sorted by name), a single
  name such as `"default"`, or a vector of names. Each becomes one CMData variable, in the
  given order.
- `include_dead::Bool=false` — count dead cells too (passed to PCMM).
- `time_scale::Real=1/60` — multiplier applied to PhysiCell save times (minutes); the default
  converts to hours.
- `obs_times` — `nothing` (default: every save time) or a vector of times, in scaled units, to
  keep. Each must match a save time to within `time_tol`; a requested time that was not saved
  is an error.
- `time_tol::Real=1e-6` — absolute tolerance, in scaled units, for matching `obs_times`.
- `sd_floor` — `nothing` (default) or a number; when given, every SD is raised to at least this
  value. Any SD that is still 0 (e.g. at t = 0, where all replicates agree) is an error, since
  the Gaussian likelihood is undefined there: drop that time with `obs_times` or set `sd_floor`.

Every monad needs at least 2 replicates. Replicates must share save times, and so must all
CM parameter sets (after `obs_times` selection).

# Returns
A `NamedTuple` with fields
- `data::CMData` — shape `[n_times × n_cell_types × 1 × n_cm_param_sets]`; parameter-set
  labels read `"name=value, …"` from the varied parameters (`"monad<id>"` if none vary).
- `cm_sample` — `CMSample` of the numeric varied parameters (`[n_cm_param_sets × n_varied]`,
  rows in the same order as `data`'s parameter sets, columns named by PCMM's short parameter
  names), or `nothing` when no numeric parameter varies (single monad or simulation vector).
- `monad_ids::Vector{Int}` — monad behind each parameter set (empty for a simulation vector).
- `simulation_ids::Vector{Vector{Int}}` — replicate simulation ids per parameter set.

Parameter sets are ordered as PCMM's `monadsTable` sorts them: by every varied parameter, the
first column as primary key. For a 2-D grid this is row-major in (first, second) parameter.

# Performance
Counts come from PCMM's per-simulation population time series. The first read of a simulation
parses every output snapshot (about 1 s per simulation); PCMM caches the result under the
simulation's `summary/` folder, so later calls are fast. PCMM has no batched reader.

# Example
```julia
using SmoreBase, PhysiCellModelManager
initializeModelManager("path/to/project")

sampling = Sampling(1)
out = cmDataFromPCMM(sampling; cell_types="default", obs_times=[24.0, 48.0, 72.0])
out.data        # CMData, [3 × 1 × 1 × n_monads]
out.cm_sample   # GridCMSample of the varied parameters
```
"""
function cmDataFromPCMM(args...; kwargs...)
    throw(ArgumentError(
        "cmDataFromPCMM requires PhysiCellModelManager.jl: run `using PhysiCellModelManager` " *
        "to load the SmoreBasePhysiCellModelManagerExt extension (got argument types " *
        "$(join(typeof.(args), ", ")))"
    ))
end

# ── PCMM-independent core (tested without PCMM) ────────────────────────────────

"""
    _cmDataFromReplicates(groups; variable_labels, cm_param_set_labels, time_scale=1/60,
                          obs_times=nothing, time_tol=1e-6, sd_floor=nothing) -> CMData

Pure core of `cmDataFromPCMM`. `groups[k]` holds the replicates of CM parameter set `k`, each a
`NamedTuple` `(id, time, counts)`: `id` names the replicate in error messages, `time` is its
vector of raw save times, `counts` its `[n_times × n_variables]` matrix.
"""
function _cmDataFromReplicates(groups::AbstractVector;
                               variable_labels::AbstractVector{<:AbstractString},
                               cm_param_set_labels::AbstractVector{<:AbstractString},
                               time_scale::Real=1 / 60,
                               obs_times::Union{Nothing,AbstractVector{<:Real}}=nothing,
                               time_tol::Real=1e-6,
                               sd_floor::Union{Nothing,Real}=nothing)
    isempty(groups) && throw(ArgumentError("no CM parameter sets to build CMData from"))
    length(cm_param_set_labels) == length(groups) || throw(ArgumentError(
        "got $(length(cm_param_set_labels)) labels for $(length(groups)) CM parameter sets"))
    if !isnothing(obs_times)
        isempty(obs_times) && throw(ArgumentError("obs_times is empty"))
        allunique(obs_times) || throw(ArgumentError("obs_times has repeated values"))
    end

    n_v = length(variable_labels)
    stats = map(zip(groups, cm_param_set_labels)) do (reps, label)
        t, Y = _stackReplicates(reps, n_v, label)
        t_scaled = Float64.(t) .* time_scale
        rows = isnothing(obs_times) ? collect(eachindex(t_scaled)) :
               [_matchTime(t_scaled, τ, time_tol, label) for τ in obs_times]
        μ, σ = _replicateMeanSD(Y[rows, :, :], sd_floor, t_scaled[rows], variable_labels, label)
        (t_scaled[rows], μ, σ)
    end

    t = stats[1][1]
    for (k, s) in enumerate(stats)
        length(s[1]) == length(t) && all(isapprox.(s[1], t; atol=time_tol)) || throw(ArgumentError(
            "CM parameter sets $(cm_param_set_labels[1]) and $(cm_param_set_labels[k]) do not " *
            "share save times; pass `obs_times` to select common times"))
    end

    μ = cat((s[2] for s in stats)...; dims=3)   # [n_times × n_variables × n_ps]
    σ = cat((s[3] for s in stats)...; dims=3)
    return CMData(μ=μ, σ=σ, times=t, variables=collect(String, variable_labels),
                  cm_param_sets=collect(String, cm_param_set_labels),
                  dim_order=[:times, :variables, :cm_param_sets])
end

# Stack one parameter set's replicates into [n_times × n_variables × n_reps], checking that they
# share save times.
function _stackReplicates(reps::AbstractVector, n_v::Int, label)
    length(reps) >= 2 || throw(ArgumentError(
        "CM parameter set $label has $(length(reps)) replicate(s); at least 2 are needed for a " *
        "sample standard deviation"))
    t = reps[1].time
    for r in reps
        r.time == t || throw(ArgumentError(
            "replicates $(reps[1].id) and $(r.id) of CM parameter set $label have different save times"))
        size(r.counts) == (length(t), n_v) || throw(ArgumentError(
            "replicate $(r.id) of CM parameter set $label has counts of size $(size(r.counts)); " *
            "expected $((length(t), n_v))"))
    end
    Y = Array{Float64}(undef, length(t), n_v, length(reps))
    for (j, r) in enumerate(reps)
        Y[:, :, j] .= r.counts
    end
    return t, Y
end

function _matchTime(t::AbstractVector{<:Real}, τ::Real, tol::Real, label)
    r = findfirst(x -> isapprox(x, τ; atol=tol), t)
    isnothing(r) && throw(ArgumentError(
        "requested time $τ is not among the save times of CM parameter set $label " *
        "(saved: $(isempty(t) ? "none" : "$(first(t)) … $(last(t)), $(length(t)) times"); " *
        "matched within time_tol=$tol)"))
    return r
end

# Mean and sample SD over the replicate axis, with optional floor and a hard error on SD = 0.
function _replicateMeanSD(Y::AbstractArray{<:Real,3}, sd_floor, t, variable_labels, label)
    μ = dropdims(mean(Y; dims=3); dims=3)
    σ = dropdims(std(Y; dims=3); dims=3)
    isnothing(sd_floor) || (σ .= max.(σ, sd_floor))
    zero_idx = findall(iszero, σ)
    isempty(zero_idx) || throw(ArgumentError(
        "replicate SD is 0 for CM parameter set $label at " *
        join(["(t=$(t[I[1]]), $(variable_labels[I[2]]))" for I in zero_idx[1:min(end, 5)]], ", ") *
        (length(zero_idx) > 5 ? ", … ($(length(zero_idx)) entries)" : "") *
        "; the Gaussian likelihood is undefined there. Drop those times with `obs_times` " *
        "or pass `sd_floor > 0`"))
    return μ, σ
end

# Build "name=value, …" labels for each row of a table of varied parameters (`columns` is a
# vector of (name, values) pairs, each values vector one entry per parameter set).
function _variedLabels(columns::AbstractVector, n::Int)
    isempty(columns) && return nothing
    return [join(["$(name)=$(vals[k])" for (name, vals) in columns], ", ") for k in 1:n]
end
