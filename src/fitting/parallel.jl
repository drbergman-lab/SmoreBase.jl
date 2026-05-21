# Fit the surrogate model for a single param_set.
# Returns (fitted_params, error_val, converged, raw_result).
function _fitOneParamSet(
    sm::AbstractSurrogateModel,
    data::AbstractCMData,
    p0_row::AbstractVector,
    prior::ParameterPrior,
    conditions::ConditionSpec,
    loss::AbstractLoss,
    optimOptions::NamedTuple,
    param_set_idx::Int,
)
    obj = _buildObjective(sm, data, conditions, loss, param_set_idx)
    opt_fn  = OptimizationFunction(obj, Optimization.AutoForwardDiff())
    lb = _lowerBounds(prior)
    ub = _upperBounds(prior)
    prob = OptimizationProblem(opt_fn, collect(Float64, p0_row), nothing; lb, ub)
    sol  = solve(prob, Fminbox(LBFGS()); optimOptions...)
    converged = !isnan(sol.objective) && !isinf(sol.objective)
    return sol.u, sol.objective, converged, sol
end

# Fit all param_sets using the provided map_fn (a resolved callable).
function _fitAllParamSets(
    sm::AbstractSurrogateModel,
    data::AbstractCMData,
    P0::AbstractMatrix,
    prior::ParameterPrior,
    conditions::ConditionSpec,
    loss::AbstractLoss,
    optimOptions::NamedTuple,
    map_fn,
)
    n_ps     = size(P0, 1)
    n_params = size(P0, 2)

    raw = map_fn(1:n_ps) do i
        _fitOneParamSet(sm, data, P0[i, :], prior, conditions, loss, optimOptions, i)
    end

    params      = Matrix{Float64}(undef, n_ps, n_params)
    errors      = Vector{Float64}(undef, n_ps)
    converged   = BitVector(undef, n_ps)
    opt_results = Vector{Any}(undef, n_ps)
    for (i, (p, e, c, r)) in enumerate(raw)
        params[i, :] = p
        errors[i]      = e
        converged[i]   = c
        opt_results[i] = r
    end

    return params, errors, converged, opt_results
end

# Resolve an executor keyword to a callable map function.
# Accepts a Symbol (:serial, :threads, :distributed) or any callable.
function _resolveExecutor(executor::Symbol)
    executor === :serial      && return map
    executor === :threads     && return _threadedMapFn()
    executor === :distributed && return _distributedMapFn()
    throw(ArgumentError(
        "Unknown executor :$executor. Valid symbols: :serial, :threads, :distributed. " *
        "Alternatively, pass any callable with signature (f, itr) -> Vector."
    ))
end
_resolveExecutor(executor) = executor  # callable passthrough

function _threadedMapFn()
    mod = _findLoadedModule("ThreadsX")
    mod === nothing && throw(ArgumentError(
        "executor=:threads requires ThreadsX: add `using ThreadsX` before calling fitSurrogate."
    ))
    return mod.map
end

function _distributedMapFn()
    mod = _findLoadedModule("Distributed")
    mod === nothing && throw(ArgumentError(
        "executor=:distributed requires Distributed: add `using Distributed` before calling fitSurrogate."
    ))
    return mod.pmap
end

function _findLoadedModule(name::String)
    for (id, mod) in Base.loaded_modules
        id.name == name && return mod
    end
    return nothing
end
