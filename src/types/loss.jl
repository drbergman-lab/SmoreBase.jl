"""
    AbstractLoss

Abstract base type for loss functions used in SM fitting.
"""
abstract type AbstractLoss end

"""
    GaussianNLL()

Gaussian negative log-likelihood loss (default).

Without full covariance (`Σ = nothing`):
`NLL = 0.5 * Σ((pred - μ)² / σ²) + 0.5 * Σ(log(2π σ²))`

With full covariance (`Σ` shape `[n_variables, n_variables, n_times, n_conditions, n_param_sets]`):
multivariate Gaussian NLL summed over time points for the given condition and param_set.
"""
struct GaussianNLL <: AbstractLoss end

"""
    CustomLoss(fn)

User-supplied loss function. `fn` is called as:
`fn(A_pred, data, param_set_idx, condition_idx) -> Float64`
"""
struct CustomLoss{F} <: AbstractLoss
    fn::F
end

"""
    _computeLoss(loss, A_pred, data, param_set_idx, condition_idx) -> Float64

Compute scalar loss between SM prediction matrix `A_pred` ([n_times × n_outputs])
and the corresponding slice of `data` at `(param_set_idx, condition_idx)`.
"""
function _computeLoss(
    ::GaussianNLL,
    A_pred::AbstractMatrix,
    data::AbstractCMData,
    pi::Int,
    ki::Int,
)
    μ_view = @view data.μ[:, :, ki, pi]
    σ_view = @view data.σ[:, :, ki, pi]
    if isnothing(data.Σ)
        return 0.5 * sum((A_pred .- μ_view) .^ 2 ./ σ_view .^ 2) +
               0.5 * sum(log.(2π .* σ_view .^ 2))
    else
        nll = 0.0
        n_t = size(A_pred, 1)
        for ti in 1:n_t
            Σ_t = @view data.Σ[:, :, ti, ki, pi]
            r = vec(A_pred[ti, :]) .- vec(μ_view[ti, :])
            nll += 0.5 * (logdet(Σ_t) + dot(r, Σ_t \ r) + size(Σ_t, 1) * log(2π))
        end
        return nll
    end
end

function _computeLoss(loss::CustomLoss, A_pred, data, pi::Int, ki::Int)
    return loss.fn(A_pred, data, pi, ki)
end
