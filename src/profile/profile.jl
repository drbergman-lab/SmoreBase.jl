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
    quantifyUncertainty(problem, fitResult, method; param_set_index) -> SMUQResult

Quantify uncertainty in fitted surrogate-model parameters.

This is the second stage of the pipeline (`fitSurrogate` → `quantifyUncertainty` →
`sampleSMPredictions`). The UQ algorithm is selected by `method::AbstractUQMethod`;
dispatch on that argument is the extension point for new UQ methods — define a
`MyMethod <: AbstractUQMethod` and add a `quantifyUncertainty(::SMFitProblem,
::SMFitResult, ::MyMethod)` method.

The `ProfileLikelihood` method computes profile likelihood UQ for one param_set,
returning a `ProfileLikelihoodResult`.

# Arguments
- `problem::SMFitProblem` — the fitting problem (model, data, prior, loss)
- `fitResult::SMFitResult` — fitted parameters from `fitSurrogate`
- `method::AbstractUQMethod` — UQ algorithm (e.g. `ProfileLikelihood()`)

# Keyword arguments
- `param_set_index::Int` — which param_set to profile (default: 1)
"""
function quantifyUncertainty(
    problem::SMFitProblem,
    fitResult::SMFitResult,
    method::ProfileLikelihood;
    param_set_index::Int = 1,
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

    p_mle      = fitResult.parameters[param_set_index, :]
    L_star     = -fitResult.errors[param_set_index]     # errors store the objective (= -LL)
    threshold  = L_star - 0.5 * quantile(Chisq(1), method.confidence_level)
    data_slice = _sliceParamSet(problem.data, param_set_index)
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
