"""
    fitSurrogate(problem, P0; executor, optimOptions) -> SMFitResult

Fit surrogate model parameters to CM summary statistics for each param_set.

One independent bounded optimization is run per param_set using `Fminbox(LBFGS())`
(via `Optimization.jl` + `OptimizationOptimJL`).

# Arguments
- `problem::SMFitProblem` — bundles the surrogate model, CM data, parameter prior, and loss function
- `P0::AbstractMatrix` — initial parameter guesses `[n_param_sets × n_sm_params]`

# Keyword arguments
- `executor` — controls how param_sets are fitted:
  - `:serial` (default) — sequential `map`
  - `:threads` — multithreaded via `ThreadsX.map` (requires `using ThreadsX`)
  - `:distributed` — distributed via `Distributed.pmap` (requires `using Distributed`)
  - Any callable `(f, itr) -> Vector` — custom executor
- `optimOptions::NamedTuple` — forwarded to `Optimization.jl` `solve()` (default: `(;)`)

# Returns
`SMFitResult` with fitted parameters in `[lb, ub]` for each param_set.

# Example
```julia
prior   = ParameterPrior([0.0, 0.0], [2.0, 10.0]; names=["r", "K"])
problem = SMFitProblem(sm, data, prior)
P0      = [0.5 5.0]
result  = fitSurrogate(problem, P0)
```
"""
function fitSurrogate(
    problem::SMFitProblem,
    P0::AbstractMatrix;
    executor                 = :serial,
    optimOptions::NamedTuple = (;),
)
    n_ps     = n_param_sets(problem.data)
    n_params = length(problem.prior)

    size(P0, 1) == n_ps ||
        throw(ArgumentError("P0 has $(size(P0, 1)) rows but data has $n_ps param_sets"))
    size(P0, 2) == n_params ||
        throw(ArgumentError("P0 has $(size(P0, 2)) columns but prior has $n_params parameters"))

    map_fn = _resolveExecutor(executor)
    params, errors, conv, opt_results = _fitAllParamSets(problem, P0, optimOptions, map_fn)

    return SMFitResult{Float64}(
        params,
        errors,
        convert(Matrix{Float64}, P0),
        problem.prior,
        conv,
        opt_results,
    )
end
