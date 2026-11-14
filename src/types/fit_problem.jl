"""
    SMFitProblem

Bundles the four ingredients of a surrogate-model fitting problem: the model,
the training data, the parameter prior, and the loss function.

Pass one `SMFitProblem` to `fitSurrogate`, `quantifyUncertainty`, and `sampleSMPredictions`
instead of threading `sm`, `data`, `prior`, and `loss` through each call.
Experimental conditions are derived automatically from the data via `_conditions`.

# Fields
- `sm` — surrogate model to fit
- `data` — CM summary statistics
- `prior` — SM parameter prior (bounds + names)
- `loss` — objective function (default: `GaussianNLL()`)

With `GaussianNLL` and no full covariance, every `σ` in a `CMData` must be positive and finite;
otherwise the constructor throws an `ArgumentError` (the likelihood would be infinite or `NaN`).
Zero or undefined SDs arise from e.g. t = 0, where all replicates agree, or a single replicate.
Drop those entries from the data, or use a `CustomLoss` that does not divide by `σ`.

# Example
```julia
problem = SMFitProblem(sm, data, prior)                        # GaussianNLL by default
problem = SMFitProblem(sm, data, prior; loss = CustomLoss(fn))
result  = fitSurrogate(problem, P0)
uq      = quantifyUncertainty(ProfileLikelihood(), problem, result, 1)   # single cm_param_set
samples = sampleSMPredictions(problem, uq)
```
"""
struct SMFitProblem
    sm::AbstractSurrogateModel
    data::AbstractCMData
    prior::ParameterPrior
    loss::AbstractLoss

    function SMFitProblem(sm::AbstractSurrogateModel, data::AbstractCMData, prior::ParameterPrior,
                          loss::AbstractLoss)
        _validateLossData(loss, data)
        return new(sm, data, prior, loss)
    end
end

SMFitProblem(sm, data, prior; loss::AbstractLoss = GaussianNLL()) =
    SMFitProblem(sm, data, prior, loss)

# Check that `data` can be scored by `loss` before any fitting starts. Default: no constraint.
_validateLossData(::AbstractLoss, ::AbstractCMData) = nothing

function _validateLossData(::GaussianNLL, data::CMData)
    isnothing(data.Σ) || return nothing
    bad = findall(s -> !(isfinite(s) && s > 0), data.σ)
    isempty(bad) && return nothing
    show_entry(I) = "(time $(I[1]), $(data.variable_labels[I[2]]), $(data.condition_labels[I[3]]), " *
                    "$(data.cm_param_set_labels[I[4]]))"
    throw(ArgumentError(
        "GaussianNLL needs every σ > 0 and finite, but $(length(bad)) entr$(length(bad) == 1 ? "y is" : "ies are") " *
        "not, e.g. σ = $(data.σ[bad[1]]) at " * join(show_entry.(bad[1:min(end, 3)]), ", ") *
        ". Drop those times (or variables) from the data, or use a CustomLoss."))
end
