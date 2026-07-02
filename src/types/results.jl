"""
    SMUQResult

Abstract base type for SM parameter uncertainty quantification results.
"""
abstract type SMUQResult end

"""
    SMFitResult{T<:Real}

Result of fitting a surrogate model to CM summary statistics.

# Fields
- `parameters` — fitted SM parameters `[n_cm_param_sets × n_sm_params]`
- `errors` — objective value (loss) per cm_param_set at the fitted parameters
- `initial_parameters` — initial guesses supplied to `fitSurrogate`
- `prior` — `ParameterPrior` used during fitting (bounds and parameter names live here)
- `converged` — convergence flag per cm_param_set
- `optim_results` — raw `Optimization.jl` solution objects
"""
struct SMFitResult{T<:Real}
    parameters::Matrix{T}
    errors::Vector{T}
    initial_parameters::Matrix{T}
    prior::ParameterPrior
    converged::BitVector
    optim_results::Vector{Any}
end

"""
    ProfileCurve{T<:Real}

Profile likelihood curve for a single SM parameter.

# Fields
- `parameter_index` — index of the profiled parameter
- `parameter_name`
- `profile_values` — swept values of the profiled parameter
- `log_likelihoods` — profile log-likelihood at each swept value
- `optimal_parameters` — full parameter vector at each grid point `[n_points × n_params]`;
  row `j` has the fixed parameter at `profile_values[j]` and all other parameters at their
  re-optimized values
- `ci_lower`, `ci_upper` — confidence interval bounds (`nothing` if profile does not cross threshold)
- `threshold` — `L* − 0.5 × χ²₁,α`
- `reference_ll` — `L*` (log-likelihood at the MLE)
"""
struct ProfileCurve{T<:Real}
    parameter_index::Int
    parameter_name::String
    profile_values::Vector{T}
    log_likelihoods::Vector{T}
    optimal_parameters::Matrix{T}
    ci_lower::Union{Nothing,T}
    ci_upper::Union{Nothing,T}
    threshold::T
    reference_ll::T
end

"""
    ProfileLikelihoodResult{T<:Real} <: SMUQResult

Result of profile likelihood UQ for one cm_param_set.

# Fields
- `profiles` — one `ProfileCurve` per SM parameter; each curve is self-contained
- `fit_result` — the `SMFitResult` used as the MLE reference (carries `prior` and initial fit)
- `times` — time grid from `CMData` (stored for use by `sampleSMPredictions`)
"""
struct ProfileLikelihoodResult{T<:Real} <: SMUQResult
    profiles::Vector{ProfileCurve{T}}
    fit_result::SMFitResult{T}
    times::Vector{T}
end

"""
    SampledPredictions{T<:Real}

SM predictions sampled within the UQ-defined parameter region.

Arrays are laid out with the sample index last so each per-sample slice is a
contiguous block in Julia's column-major memory.

# Fields
- `parameters` — sampled SM parameters `[n_sm_params × nSamples]`
- `predictions` — SM predictions `[n_times × n_outputs × nSamples]`
- `times` — time grid used for SM evaluation (`nothing` if no time axis)
"""
struct SampledPredictions{T<:Real}
    parameters::Matrix{T}
    predictions::Array{T,3}
    times::Union{Nothing,Vector{T}}
end
