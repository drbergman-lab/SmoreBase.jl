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
`fn(A_pred, data_slice::AbstractCMDataSlice, condition_idx) -> Float64`

`data_slice` is a single-param-set view (see `CMDataSlice`); `condition_idx` is the
index into `data_slice`'s condition axis for the current evaluation.
"""
struct CustomLoss{F} <: AbstractLoss
    fn::F
end

"""
    _computeLoss(loss, A_pred, data_slice, condition_idx) -> Float64

Compute scalar loss between SM prediction matrix `A_pred` (`[n_times × n_variables]`)
and the corresponding condition slice of `data_slice` at `condition_idx`.

`data_slice` must be an `AbstractCMDataSlice` — a single-param-set view produced by
`_sliceParamSet`. The `param_set_idx` dimension has already been dropped.
"""
function _computeLoss(
    ::GaussianNLL,
    A_pred::AbstractMatrix,
    data::AbstractCMDataSlice,
    ki::Int,
)
    μ_view = @view _mean(data)[:, :, ki]
    σ_view = @view _sd(data)[:, :, ki]
    Σ = _cov(data)
    if isnothing(Σ)
        return 0.5 * sum((A_pred .- μ_view) .^ 2 ./ σ_view .^ 2) +
               0.5 * sum(log.(2π .* σ_view .^ 2))
    else
        nll = 0.0
        n_t = size(A_pred, 1)
        for ti in 1:n_t
            Σ_t = @view Σ[:, :, ti, ki]
            r = vec(A_pred[ti, :]) .- vec(μ_view[ti, :])
            nll += 0.5 * (logdet(Σ_t) + dot(r, Σ_t \ r) + size(Σ_t, 1) * log(2π))
        end
        return nll
    end
end

function _computeLoss(loss::CustomLoss, A_pred, data::AbstractCMDataSlice, ki::Int)
    return loss.fn(A_pred, data, ki)
end
