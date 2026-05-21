"""
    ParameterPrior(distributions, names)
    ParameterPrior(lower, upper; names)

Prior distributions over SM parameters — one `UnivariateDistribution` per parameter.
Box bounds are represented as `Uniform(lb, ub)`.

Optimization bounds are derived via `_lowerBounds` and `_upperBounds`.

# Examples
```julia
# Box bounds shorthand
prior = ParameterPrior([0.0, 0.0], [2.0, 10.0]; names=["r", "K"])

# Custom priors
prior = ParameterPrior(
    [truncated(Normal(1.0, 0.5), 0.0, 2.0), Uniform(1.0, 20.0)],
    ["r", "K"],
)
```
"""
struct ParameterPrior
    distributions::Vector{<:UnivariateDistribution}
    names::Vector{String}
end

function ParameterPrior(
    lower::AbstractVector,
    upper::AbstractVector;
    names::Vector{String} = ["p$i" for i in eachindex(lower)],
)
    length(lower) == length(upper) ||
        throw(ArgumentError("lower and upper must have the same length"))
    all(lower .< upper) ||
        throw(ArgumentError("all lower bounds must be strictly less than upper bounds"))
    return ParameterPrior(
        UnivariateDistribution[Uniform(Float64(l), Float64(u)) for (l, u) in zip(lower, upper)],
        names,
    )
end

_lowerBounds(prior::ParameterPrior) = Float64[minimum(d) for d in prior.distributions]
_upperBounds(prior::ParameterPrior) = Float64[maximum(d) for d in prior.distributions]

Base.length(prior::ParameterPrior) = length(prior.distributions)
