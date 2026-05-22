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
    _uq(sm, data, fitResult, method; conditions, param_set_index) -> ProfileLikelihoodResult

Compute profile likelihood UQ for one param_set.

Not intended to be called by end users directly; a public wrapper will be added
once the higher-level pipeline API is designed.
"""
function _uq(
    sm::AbstractSurrogateModel,
    data::AbstractCMData,
    fitResult::SMFitResult,
    method::ProfileLikelihood;
    conditions::ConditionSpec = ConditionSpec(),
    param_set_index::Int      = 1,
)
    n_params = length(fitResult.prior)
    lb = !isnothing(method.bounds) ? _lowerBounds(method.bounds) : _lowerBounds(fitResult.prior)
    ub = !isnothing(method.bounds) ? _upperBounds(method.bounds) : _upperBounds(fitResult.prior)

    bad = findall(i -> isinf(lb[i]) || isinf(ub[i]), 1:n_params)
    if !isempty(bad)
        names_str = join([fitResult.prior.names[i] for i in bad], ", ")
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
    data_slice = _sliceParamSet(data, param_set_index)

    profiles = Vector{ProfileCurve{Float64}}(undef, n_params)

    for i in 1:n_params
        mle_val = p_mle[i]

        # Proportional split: allocate grid points in proportion to distance from MLE to each boundary
        frac_left = (mle_val - lb[i]) / (ub[i] - lb[i])
        n_left    = max(1, round(Int, frac_left * (method.n_points - 1)) + 1)
        n_right   = method.n_points - n_left + 1   # includes the shared MLE point

        left_grid  = collect(range(lb[i],   mle_val; length = n_left))
        right_grid = collect(range(mle_val, ub[i];   length = n_right))
        grid       = [left_grid; right_grid[2:end]]  # deduplicate MLE point → n_points total

        lls        = Vector{Float64}(undef, method.n_points)
        opt_params = Matrix{Float64}(undef, method.n_points, n_params)

        # Evaluate the MLE grid point (index n_left)
        lls[n_left], opt_params[n_left, :] =
            _profileLL(sm, data_slice, copy(p_mle), conditions, i, mle_val, lb, ub)

        # Left scan: outward from MLE toward lb
        p_warm = opt_params[n_left, :]
        for j in (n_left - 1):-1:1
            lls[j], opt_params[j, :] =
                _profileLL(sm, data_slice, p_warm, conditions, i, grid[j], lb, ub)
            p_warm = opt_params[j, :]
        end

        # Right scan: outward from MLE toward ub
        p_warm = opt_params[n_left, :]
        for j in (n_left + 1):method.n_points
            lls[j], opt_params[j, :] =
                _profileLL(sm, data_slice, p_warm, conditions, i, grid[j], lb, ub)
            p_warm = opt_params[j, :]
        end

        ci_lower, ci_upper = _computeCI(grid, lls, threshold)

        profiles[i] = ProfileCurve{Float64}(
            i,
            fitResult.prior.names[i],
            grid,
            lls,
            opt_params,
            ci_lower,
            ci_upper,
            threshold,
            L_star,
        )
    end

    return ProfileLikelihoodResult{Float64}(profiles, fitResult, data.times)
end

# Compute the profile log-likelihood at a fixed value `v` for parameter `fixed_idx`.
# Re-optimizes all remaining parameters within their bounds.
function _profileLL(
    sm::AbstractSurrogateModel,
    data_slice::AbstractCMDataSlice,
    p_init::Vector{Float64},
    conditions::ConditionSpec,
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
        obj = _buildObjective(sm, data_slice, conditions, GaussianNLL())
        return -obj(p_full, nothing), p_full
    end

    p_init_free = p_init[free_idx]
    lb_free     = lb[free_idx]
    ub_free     = ub[free_idx]

    full_obj = _buildObjective(sm, data_slice, conditions, GaussianNLL())

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
