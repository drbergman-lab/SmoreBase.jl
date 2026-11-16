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

With full covariance (`Σ` shape `[n_variables, n_variables, n_times, n_conditions, n_cm_param_sets]`):
multivariate Gaussian NLL summed over time points for the given condition and cm_param_set.
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
`_sliceCmParamSet`. The `cm_param_set_idx` dimension has already been dropped.
"""
function _computeLoss(
    ::GaussianNLL,
    A_pred::AbstractMatrix,
    data::AbstractCMDataSlice,
    ki::Int,
)
    Σ = _cov(data)
    if isnothing(Σ)
        return _gaussianNLLIndependent(A_pred, _mean(data), _sd(data), ki)
    else
        μ_view = @view _mean(data)[:, :, ki]
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

# Independent-noise Gaussian NLL for condition `ki`:
# `0.5 * Σ (A_pred - μ)² / σ² + 0.5 * Σ log(2π σ²)`, in one column-major pass (times fastest,
# then variables) with no temporary arrays. The squared-residual accumulator takes `A_pred`'s
# element type, so ForwardDiff `Dual`s pass through during fitting.
function _gaussianNLLIndependent(A_pred::AbstractMatrix, μ::AbstractArray, σ::AbstractArray, ki::Int)
    rows, cols = axes(A_pred)
    (rows, cols) == (axes(μ, 1), axes(μ, 2)) == (axes(σ, 1), axes(σ, 2)) || throw(DimensionMismatch(
        "prediction has size $(size(A_pred)), but the data's mean and sd have " *
        "$((size(μ, 1), size(μ, 2))) and $((size(σ, 1), size(σ, 2))) per condition"
    ))
    checkbounds(μ, rows, cols, ki)
    checkbounds(σ, rows, cols, ki)
    sq = zero(float(promote_type(eltype(A_pred), eltype(μ), eltype(σ))))
    lg = zero(float(eltype(σ)))
    @inbounds for j in cols, i in rows
        s   = σ[i, j, ki]
        sq += (A_pred[i, j] - μ[i, j, ki])^2 / s^2
        lg += log(2π * s^2)
    end
    return 0.5 * sq + 0.5 * lg
end

function _computeLoss(loss::CustomLoss, A_pred, data::AbstractCMDataSlice, ki::Int)
    return loss.fn(A_pred, data, ki)
end
