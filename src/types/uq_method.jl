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
