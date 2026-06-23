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

# ── SM parameter sampling ──────────────────────────────────────────────────────
#
# `sampleSMParameters` is the public, per-UQ-result extension point: it maps an
# `SMUQResult` to a matrix of drawn SM-parameter vectors [n_sm_params × nSamples].
# `sampleSMPredictions` calls it, so adding a method here is all a new UQ result
# type needs in order to flow through the prediction-sampling stage.
#
# Profile draws go through `_sampleFromProfiles`, which samples each parameter independently
# from its marginal profile curve. `sampleSMParametersInBounds` covers the uniform-box case
# (a flat profile reduces the inverse-CDF draw to the affine map onto [lb, ub]) in closed
# form; both share `_sobolUnit` for the underlying low-discrepancy points.

# Shared unit-hypercube point generator: a [n_params × nSamples] low-discrepancy Sobol
# grid in [0,1] with a per-dimension Cranley-Patterson shift. Both the profile sampler
# and the bounds sampler draw from this, so they consume the RNG identically.
function _sobolUnit(n_params::Int, nSamples::Int, rng::AbstractRNG)
    U     = QuasiMonteCarlo.sample(nSamples, zeros(n_params), ones(n_params), SobolSample())
    shift = rand(rng, n_params)           # Cranley-Patterson shift: one uniform offset per dimension
    return mod.(U .+ shift, 1.0)         # wrap to [0,1] — preserves low-discrepancy structure
end

# Core sampler: draw [n_sm_params × nSamples] from a set of marginal profile curves.
# Each parameter is drawn independently from its profile-LL-weighted marginal via
# piecewise-linear inverse-CDF sampling, driven by the shared Sobol points.
function _sampleFromProfiles(
    profiles::AbstractVector{<:ProfileCurve},
    nSamples::Int,
    rng::AbstractRNG,
)
    n_params = length(profiles)
    U        = _sobolUnit(n_params, nSamples, rng)

    params = Matrix{Float64}(undef, n_params, nSamples)
    for (i, pc) in enumerate(profiles)
        params[i, :] = _applyProfileInverseCDF(pc.profile_values, pc.log_likelihoods, U[i, :])
    end
    return params
end

"""
    sampleSMParameters(uqResult::SMUQResult; nSamples, rng) -> Matrix

Draw SM-parameter vectors from the distribution encoded in `uqResult`, returned as
`[n_sm_params × nSamples]`.

This is the extension point for custom UQ methods: to make a new
`MyResult <: SMUQResult` flow through [`sampleSMPredictions`](@ref), add a method

    SmoreBase.sampleSMParameters(r::MyResult; nSamples, rng) = ...

# Method — `ProfileLikelihoodResult`
Each parameter is sampled independently from its marginal profile-LL distribution via
piecewise-linear inverse-CDF sampling. This is a product-measure approximation: marginal
LL shapes are respected, but correlations between parameters are ignored. Low-discrepancy
Sobol points with a Cranley-Patterson shift drive the inverse CDF.

# Keyword arguments
- `nSamples::Int` — number of parameter vectors to draw (default: 100)
- `rng` — random number generator (default: `Random.default_rng()`)
"""
function sampleSMParameters(
    uq::SMUQResult;
    nSamples::Int=100,
    rng::AbstractRNG=Random.default_rng(),
)
    throw(ArgumentError(
        "No `sampleSMParameters` method is defined for $(typeof(uq)). " *
        "Define `SmoreBase.sampleSMParameters(r::MyResult; nSamples, rng)` for your custom `MyResult <: SMUQResult`."
    ))
end

sampleSMParameters(
    uq::ProfileLikelihoodResult;
    nSamples::Int    = 100,
    rng::AbstractRNG = Random.default_rng(),
) = _sampleFromProfiles(uq.profiles, nSamples, rng)

# Time grid carried by a UQ result, used by `sampleSMPredictions` to evaluate the SM.
# Defaults to the `times` field; a custom `SMUQResult` without that field can override.
function _predictionTimes(uqResult::SMUQResult)
    if hasproperty(uqResult, :times)
        return getproperty(uqResult, :times)
    end
    throw(ArgumentError(
        "UQ result $(typeof(uqResult)) must provide a `times` field or overload `_predictionTimes(::$(typeof(uqResult)))`."
    ))
end

"""
    sampleSMParametersInBounds(lb, ub; nSamples, rng) -> Matrix

Draw `nSamples` SM-parameter vectors uniformly within the box `[lb, ub]`, returned as
`[n_sm_params × nSamples]`.

This is the uninformative special case of [`sampleSMParameters`](@ref): a flat profile over
the box yields uniform marginals, for which the inverse-CDF draw reduces in closed form to
the affine map `lb + u·(ub − lb)`. So this samples directly from the bounds — sharing the
Sobol + Cranley-Patterson point generation with the profile sampler but constructing no
intermediate curve objects, keeping it allocation-light on hot paths. The result is
bit-identical to routing flat curves through the profile sampler (asserted in the tests).

Intended for callers (e.g. SmoreGSA) that need uniform SM-parameter samples within an
interpolated CI box without reaching into SmoreBase internals.

# Keyword arguments
- `nSamples::Int` — number of parameter vectors to draw (default: 100)
- `rng` — random number generator (default: `Random.default_rng()`)
"""
function sampleSMParametersInBounds(
    lb::AbstractVector,
    ub::AbstractVector;
    nSamples::Int    = 100,
    rng::AbstractRNG = Random.default_rng(),
)
    nSamples >= 1 || throw(ArgumentError("nSamples must be >= 1 (got $nSamples)."))
    length(lb) == length(ub) ||
        throw(ArgumentError("lb and ub must have the same length (got $(length(lb)) and $(length(ub)))."))
    all(lb .<= ub) ||
        throw(ArgumentError("each lb[i] must be <= ub[i]; an interpolated CI box may have crossed."))
    U = _sobolUnit(length(lb), nSamples, rng)
    return lb .+ U .* (ub .- lb)         # closed form of the flat-profile inverse CDF
end

"""
    sampleSMPredictions(problem, uqResult; nSamples, rng) -> SampledPredictions

Sample SM predictions by drawing parameter vectors from the distribution encoded in
`uqResult` and evaluating the SM at each draw.

Accepts any `uqResult::SMUQResult`. The parameter draws are delegated to
[`sampleSMParameters`](@ref), which dispatches on the concrete result type, so this
function works unchanged for any UQ method that implements that hook. For
`ProfileLikelihoodResult`, each parameter is sampled independently from its marginal
profile-LL distribution via piecewise-linear inverse-CDF sampling (a product-measure
approximation — marginal LL shapes are respected, correlations are ignored).

# Arguments
- `problem::SMFitProblem` — bundles the surrogate model and data (conditions derived from data)
- `uqResult::SMUQResult` — UQ result from `quantifyUncertainty`

# Keyword arguments
- `nSamples::Int` — number of parameter samples (default: 100)
- `rng` — random number generator (default: `Random.default_rng()`)

# Returns
`SampledPredictions` with:
- `parameters` — `[n_sm_params × nSamples]`
- `predictions` — `[n_times × n_outputs × nSamples]`
- `times` — the time grid used for evaluation (from the UQ result)

# Example
```julia
samples = sampleSMPredictions(problem, uq_result; nSamples=200)
```
"""
function sampleSMPredictions(
    problem::SMFitProblem,
    uqResult::SMUQResult;
    nSamples::Int    = 100,
    rng::AbstractRNG = Random.default_rng(),
)
    nSamples >= 1 || throw(ArgumentError("nSamples must be >= 1 (got $nSamples)."))
    params = sampleSMParameters(uqResult; nSamples = nSamples, rng = rng)

    # Evaluate SM at each sample (first condition only in v0)
    times      = _predictionTimes(uqResult)
    cond_label = _conditions(problem.data)[1]
    A_test     = _evaluate(problem.sm, times, params[:, 1], cond_label)
    n_t, n_out = size(A_test)

    preds = Array{Float64,3}(undef, n_t, n_out, nSamples)
    preds[:, :, 1] = A_test
    for s in 2:nSamples
        preds[:, :, s] = _evaluate(problem.sm, times, params[:, s], cond_label)
    end

    return SampledPredictions{Float64}(params, preds, times)
end
