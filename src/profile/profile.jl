"""
    AbstractUQMethod

Abstract base type for SM parameter uncertainty quantification methods.
"""
abstract type AbstractUQMethod end

"""
    ProfileLikelihood(; n_points, confidence_level, bounds)

Profile likelihood UQ method.

For each SM parameter, builds a grid of `n_points` values **anchored at the MLE**,
fixes that parameter, re-optimizes all remaining parameters, and records the
log-likelihood at each grid point. Confidence intervals are computed via Wilks' theorem:

    CI = {θᵢ : PL(θᵢ) ≥ L* − 0.5 × χ²₁,α}

where `L*` is the log-likelihood at the MLE and `χ²₁,α = quantile(Chisq(1), α)`.

The grid is split proportionally: points are allocated to each side of the MLE in
proportion to the distance from the MLE to each boundary. Each half is scanned
outward from the MLE so the inner optimizer always warm-starts near the peak.

# Fields
- `n_points` — number of grid points per parameter profile (default: 50)
- `confidence_level` — confidence level α (default: 0.95 → threshold ≈ L* − 1.92)
- `bounds` — profile range; if `nothing`, uses the `SMFitResult` optimization bounds
"""
struct ProfileLikelihood <: AbstractUQMethod
    n_points::Int
    confidence_level::Float64
    bounds::Union{Nothing,ParameterPrior}
end

ProfileLikelihood(;
    n_points::Int            = 50,
    confidence_level::Float64 = 0.95,
    bounds                   = nothing,
) = ProfileLikelihood(n_points, confidence_level, bounds)

"""
    quantifyUncertainty(method, problem, fitResult; executor) -> Vector{ProfileLikelihoodResult}
    quantifyUncertainty(method, problem, fitResult, cm_param_set_index::Integer) -> ProfileLikelihoodResult
    quantifyUncertainty(method, problem, fitResult, cm_param_set_indices::AbstractVector{<:Integer}; executor) -> Vector{ProfileLikelihoodResult}

Quantify uncertainty in fitted surrogate-model parameters.

This is the second stage of the pipeline (`fitSurrogate` → `quantifyUncertainty` →
`sampleSMPredictions`). The UQ algorithm is selected by `method::AbstractUQMethod`. To add a
new method, define `MyMethod <: AbstractUQMethod` and a `quantifyUncertainty(::MyMethod,
::SMFitProblem, ::SMFitResult, ...)` method.

Three methods, each with a fixed return type (no runtime branching on argument value within a
single method):
- No 4th argument: profiles **all** cm_param_sets in `problem.data`, row-aligned, and always
  returns a `Vector` (even when there is only one cm_param_set) — the default entry point for
  workflows with more than one cm_param_set (SmoreGSA, SmoreFit).
- `cm_param_set_index::Integer`: profiles one cm_param_set, returning a bare `ProfileLikelihoodResult`
  — opt in explicitly when you only want a single result.
- `cm_param_set_indices::AbstractVector{<:Integer}`: profiles an explicit subset/order of
  cm_param_sets; the no-argument form delegates here with `1:n_cm_param_sets(problem.data)`.

The `ProfileLikelihood` method computes profile likelihood UQ via Wilks' theorem.

# Arguments
- `method::AbstractUQMethod` — UQ algorithm (e.g. `ProfileLikelihood()`)
- `problem::SMFitProblem` — the fitting problem (model, data, prior, loss)
- `fitResult::SMFitResult` — fitted parameters from `fitSurrogate`; if omitted, computed
  internally via `fitSurrogate(problem; executor)` (default `P0` = prior medians) and embedded
  in the returned result(s)' `fit_result` field

# Keyword arguments
- `executor` — controls how cm_param_sets are profiled (same semantics as `fitSurrogate`):
  `:serial` (default), `:threads`, `:distributed`, or any callable `(f, itr) -> Vector`.
  Not accepted by the single-`cm_param_set_index` method (with or without `fitResult`). When
  `fitResult` is omitted, `executor` also controls the internal `fitSurrogate` call — for the
  all-cm_param_sets and explicit-subset forms only; the single-index fitResult-free form always
  fits with the default `:serial` executor.

# Example
```julia
uq_all = quantifyUncertainty(ProfileLikelihood(), problem, result)        # all cm_param_sets
uq_one = quantifyUncertainty(ProfileLikelihood(), problem, result, 1)     # just the first
uq_fit = quantifyUncertainty(ProfileLikelihood(), problem)                # fits internally
```
"""
function quantifyUncertainty(method::ProfileLikelihood, problem::SMFitProblem, fitResult::SMFitResult;
                              executor = :serial)
    return quantifyUncertainty(method, problem, fitResult, 1:n_cm_param_sets(problem.data); executor)
end

function quantifyUncertainty(method::ProfileLikelihood, problem::SMFitProblem, fitResult::SMFitResult,
                              cm_param_set_indices::AbstractVector{<:Integer}; executor = :serial)
    map_fn = _resolveExecutor(executor)
    return Vector{ProfileLikelihoodResult{Float64}}(
        map_fn(i -> quantifyUncertainty(method, problem, fitResult, i), cm_param_set_indices)
    )
end

"""
    quantifyUncertainty(method, problem::SMFitProblem; executor) -> Vector{ProfileLikelihoodResult}
    quantifyUncertainty(method, problem::SMFitProblem, cm_param_set_index::Integer) -> ProfileLikelihoodResult
    quantifyUncertainty(method, problem::SMFitProblem, cm_param_set_indices::AbstractVector{<:Integer}; executor) -> Vector{ProfileLikelihoodResult}

`fitResult`-free forms: fit `problem` first via `fitSurrogate(problem; executor)` (default `P0`
= prior medians), then profile the resulting MLE. The fitted `SMFitResult` is not discarded — it
is embedded in each returned `ProfileLikelihoodResult.fit_result`.

# Example
```julia
uq_all = quantifyUncertainty(ProfileLikelihood(), problem)        # fits internally, all cm_param_sets
uq_one = quantifyUncertainty(ProfileLikelihood(), problem, 1)     # fits internally, just the first
```
"""
function quantifyUncertainty(method::ProfileLikelihood, problem::SMFitProblem; executor = :serial)
    fitResult = fitSurrogate(problem; executor)
    return quantifyUncertainty(method, problem, fitResult; executor)
end

function quantifyUncertainty(method::ProfileLikelihood, problem::SMFitProblem,
                              cm_param_set_index::Integer)
    fitResult = fitSurrogate(problem)
    return quantifyUncertainty(method, problem, fitResult, cm_param_set_index)
end

function quantifyUncertainty(method::ProfileLikelihood, problem::SMFitProblem,
                              cm_param_set_indices::AbstractVector{<:Integer}; executor = :serial)
    fitResult = fitSurrogate(problem; executor)
    return quantifyUncertainty(method, problem, fitResult, cm_param_set_indices; executor)
end

function quantifyUncertainty(
    method::ProfileLikelihood,
    problem::SMFitProblem,
    fitResult::SMFitResult,
    cm_param_set_index::Integer,
)
    n_params = length(problem.prior)
    lb = !isnothing(method.bounds) ? _lowerBounds(method.bounds) : _lowerBounds(problem.prior)
    ub = !isnothing(method.bounds) ? _upperBounds(method.bounds) : _upperBounds(problem.prior)

    bad = findall(i -> isinf(lb[i]) || isinf(ub[i]), 1:n_params)
    if !isempty(bad)
        names_str = join([problem.prior.names[i] for i in bad], ", ")
        @warn "Profile sweep range is unbounded for parameter(s): $names_str. " *
              "Pass explicit `bounds::ParameterPrior` to `ProfileLikelihood` to set finite sweep limits."
        throw(ArgumentError(
            "Cannot build profile grid: parameter(s) [$names_str] have infinite sweep bounds. " *
            "Provide finite bounds via `ProfileLikelihood(bounds = ParameterPrior(lower, upper))`."
        ))
    end

    p_mle      = fitResult.parameters[cm_param_set_index, :]
    L_star     = -fitResult.errors[cm_param_set_index]     # errors store the objective (= -LL)
    threshold  = L_star - 0.5 * quantile(Chisq(1), method.confidence_level)
    data_slice = _sliceCmParamSet(problem.data, cm_param_set_index)
    conditions = _conditions(problem.data)

    profiles = Vector{ProfileCurve{Float64}}(undef, n_params)

    for i in 1:n_params
        mle_val = p_mle[i]

        grid, center_idx = if mle_val == lb[i]
            @assert method.n_points > 1 "Profile grid must have at least 2 points to scan away from MLE. Only $(method.n_points) requested."
            collect(range(mle_val, ub[i]; length = method.n_points)), 1
        elseif mle_val == ub[i]
            @assert method.n_points > 1 "Profile grid must have at least 2 points to scan away from MLE. Only $(method.n_points) requested."
            collect(range(lb[i], mle_val; length=method.n_points)), method.n_points
        else
            @assert method.n_points > 2 "Profile grid must have at least 3 points to scan away from MLE in two directions. Only $(method.n_points) requested."
            # Proportional split: allocate grid points in proportion to distance from MLE to each boundary
            frac_left = (mle_val - lb[i]) / (ub[i] - lb[i])
            n_left = min(method.n_points - 2, # leave at least 2 points for the right side (including MLE)
                max(1, # ensure at least 1 point on the left side (excluding MLE)
                    round(Int, frac_left * (method.n_points - 1))
                )
            ) # excluding the MLE point, how many points to the left
            n_right = method.n_points - n_left - 1   # excluding the MLE point, how many points to the right

            left_grid = collect(range(lb[i], mle_val; length=n_left + 1))  # include MLE point
            right_grid = collect(range(mle_val, ub[i]; length=n_right + 1))  # include MLE point
            [left_grid; right_grid[2:end]], n_left + 1  # deduplicate MLE point → n_points total
        end

        lls        = Vector{Float64}(undef, method.n_points)
        opt_params = Matrix{Float64}(undef, method.n_points, n_params)

        # Evaluate the MLE grid point
        lls[center_idx], opt_params[center_idx, :] =
            _profileLL(problem.sm, data_slice, copy(p_mle), conditions, problem.loss, i, mle_val, lb, ub)

        # Left scan: outward from MLE toward lb
        p_warm = opt_params[center_idx, :]
        for j in (center_idx-1):-1:1
            lls[j], opt_params[j, :] =
                _profileLL(problem.sm, data_slice, p_warm, conditions, problem.loss, i, grid[j], lb, ub)
            p_warm = opt_params[j, :]
        end

        # Right scan: outward from MLE toward ub
        p_warm = opt_params[center_idx, :]
        for j in (center_idx + 1):method.n_points
            lls[j], opt_params[j, :] =
                _profileLL(problem.sm, data_slice, p_warm, conditions, problem.loss, i, grid[j], lb, ub)
            p_warm = opt_params[j, :]
        end

        ci_lower, ci_upper = _computeCI(grid, lls, threshold)

        profiles[i] = ProfileCurve{Float64}(
            i,
            problem.prior.names[i],
            grid,
            lls,
            opt_params,
            ci_lower,
            ci_upper,
            threshold,
            L_star,
        )
    end

    return ProfileLikelihoodResult{Float64}(profiles, fitResult, _times(problem.data))
end

# Compute the profile log-likelihood at a fixed value `v` for parameter `fixed_idx`.
# Re-optimizes all remaining parameters within their bounds.
function _profileLL(
    sm::AbstractSurrogateModel,
    data_slice::AbstractCMDataSlice,
    p_init::Vector{Float64},
    conditions::ConditionSpec,
    loss::AbstractLoss,
    fixed_idx::Int,
    fixed_val::Float64,
    lb::Vector{Float64},
    ub::Vector{Float64},
)
    n_params = length(p_init)

    # Indices of free (non-fixed) parameters
    free_idx = setdiff(1:n_params, fixed_idx)

    # If all parameters are fixed (n_params == 1), evaluate directly
    if isempty(free_idx)
        p_full = copy(p_init)
        p_full[fixed_idx] = fixed_val
        obj = _buildObjective(sm, data_slice, conditions, loss)
        return -obj(p_full, nothing), p_full
    end

    p_init_free = p_init[free_idx]
    lb_free     = lb[free_idx]
    ub_free     = ub[free_idx]

    full_obj = _buildObjective(sm, data_slice, conditions, loss)

    function reduced_obj(p_free, _hyper)
        T = eltype(p_free)
        p_full = Vector{T}(undef, n_params)
        p_full[free_idx]  = p_free
        p_full[fixed_idx] = T(fixed_val)
        return full_obj(p_full, nothing)
    end

    opt_fn = OptimizationFunction(reduced_obj, Optimization.AutoForwardDiff())
    prob   = OptimizationProblem(opt_fn, collect(Float64, p_init_free), nothing;
                                 lb = lb_free, ub = ub_free)
    sol    = solve(prob, Fminbox(LBFGS()))

    p_full = Vector{Float64}(undef, n_params)
    p_full[free_idx]  = sol.u
    p_full[fixed_idx] = fixed_val
    return -sol.objective, p_full
end
