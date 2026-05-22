# Apply the inverse CDF of a profile LL curve to a vector of pre-drawn u-values in [0,1].
# Builds a trapezoid-rule CDF from exp(ll) weights, then inverts by piecewise-linear
# interpolation. Separating CDF application from u-value generation lets the caller use
# any sampling scheme (iid, Sobol, LHS) for the unit-hypercube points.
function _applyProfileInverseCDF(
    profile_values  :: AbstractVector,
    log_likelihoods :: AbstractVector,
    u_values        :: AbstractVector,
)
    xs = profile_values
    w  = exp.(log_likelihoods .- maximum(log_likelihoods))

    m   = length(xs)
    cdf = Vector{Float64}(undef, m)
    cdf[1] = 0.0
    for i in 2:m
        cdf[i] = cdf[i-1] + 0.5 * (w[i-1] + w[i]) * (xs[i] - xs[i-1])
    end
    cdf ./= cdf[end]

    samples = Vector{Float64}(undef, length(u_values))
    for k in eachindex(u_values)
        u = u_values[k]
        i = searchsortedlast(cdf, u)
        i = clamp(i, 1, m - 1)
        Δc = cdf[i+1] - cdf[i]
        t  = Δc < eps(Float64) ? 0.0 : (u - cdf[i]) / Δc
        samples[k] = xs[i] + t * (xs[i+1] - xs[i])
    end
    return samples
end

# ── _sampleSMParams ────────────────────────────────────────────────────────────
#
# Unified SM parameter sampler. Returns [n_sm_params × n].
# Two dispatch methods cover the two call sites in the pipeline:
#   1. ProfileLikelihoodResult  — used by sampleSMPredictions (LL-weighted marginal CDFs)
#   2. (lb, ub)                 — used by SmoreGSA (uniform LHS within an interpolated CI box)
#
# Both reduce to inverse-CDF sampling in [0,1]^n_sm_params; the difference is the CDF:
# method 1 uses the empirical LL-weighted CDF, method 2 uses a uniform CDF on the box.
# A future improvement to method 2 would interpolate the full profile LL curves across
# the CM parameter grid and use the same weighted CDF as method 1 — see the comment in
# SmoreGSA's _buildCMCallable for details.

function _sampleSMParams(
    uq  :: ProfileLikelihoodResult,
    n   :: Int,
    rng :: AbstractRNG,
)
    n_params = length(uq.profiles)

    U     = QuasiMonteCarlo.sample(n, zeros(n_params), ones(n_params), SobolSample())  # [n_params × n] in [0,1]
    shift = rand(rng, n_params)           # Cranley-Patterson shift: one uniform offset per dimension
    U     = mod.(U .+ shift, 1.0)        # wrap to [0,1] — preserves low-discrepancy structure

    params = Matrix{Float64}(undef, n_params, n)
    for (i, pc) in enumerate(uq.profiles)
        params[i, :] = _applyProfileInverseCDF(pc.profile_values, pc.log_likelihoods, U[i, :])
    end
    return params
end

function _sampleSMParams(
    lb  :: AbstractVector,
    ub  :: AbstractVector,
    n   :: Int,
    rng :: AbstractRNG,
)
    n_params = length(lb)
    U     = QuasiMonteCarlo.sample(n, zeros(n_params), ones(n_params), SobolSample())  # [n_params × n] in [0,1]
    shift = rand(rng, n_params)           # Cranley-Patterson shift: one uniform offset per dimension
    U     = mod.(U .+ shift, 1.0)        # wrap to [0,1] — preserves low-discrepancy structure
    return lb .+ U .* (ub .- lb)         # linear map to [lb, ub]
end

"""
    sampleSMPredictions(sm, uqResult; nSamples, conditions, rng) -> SampledPredictions

Sample SM predictions by drawing parameter vectors from the distribution encoded in
`uqResult` and evaluating the SM at each draw.

For `ProfileLikelihoodResult`: each parameter is sampled independently from its marginal
profile-LL distribution using piecewise-linear inverse-CDF sampling. The unnormalized
density at each grid point is `exp(ll - max(ll))`; a trapezoid-rule CDF interpolates
between grid points so that samples are continuous rather than pinned to discrete grid
locations. This is a product-measure approximation — marginal LL shapes are respected but
correlations between parameters are ignored.

# Arguments
- `sm` — the fitted surrogate model
- `uqResult::ProfileLikelihoodResult` — UQ result from `_uq`

# Keyword arguments
- `nSamples::Int` — number of parameter samples (default: 100)
- `conditions::ConditionSpec` — conditions at which to evaluate the SM (default: `ConditionSpec()`)
- `rng` — random number generator (default: `Random.default_rng()`)

# Returns
`SampledPredictions` with:
- `parameters` — `[n_sm_params × nSamples]`
- `predictions` — `[n_times × n_outputs × nSamples]`
- `times` — the time grid used for evaluation (copied from `uqResult.times`)

# Example
```julia
samples = sampleSMPredictions(sm, uq_result; nSamples=200)
```
"""
function sampleSMPredictions(
    sm::AbstractSurrogateModel,
    uqResult::ProfileLikelihoodResult;
    nSamples::Int             = 100,
    conditions::ConditionSpec = ConditionSpec(),
    rng::AbstractRNG          = Random.default_rng(),
)
    params = _sampleSMParams(uqResult, nSamples, rng)

    # Evaluate SM at each sample (first condition only in v0)
    times      = uqResult.times
    cond_label = conditions[1]
    A_test     = _evaluate(sm, times, params[:, 1], cond_label)
    n_t, n_out = size(A_test)

    preds = Array{Float64,3}(undef, n_t, n_out, nSamples)
    preds[:, :, 1] = A_test
    for s in 2:nSamples
        preds[:, :, s] = _evaluate(sm, times, params[:, s], cond_label)
    end

    return SampledPredictions{Float64}(params, preds, times)
end
