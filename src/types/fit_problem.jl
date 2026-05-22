"""
    SMFitProblem

Bundles the four ingredients of a surrogate-model fitting problem: the model,
the training data, the parameter prior, and the loss function.

Pass one `SMFitProblem` to `fitSurrogate`, `_uq`, and `sampleSMPredictions`
instead of threading `sm`, `data`, `prior`, and `loss` through each call.
Experimental conditions are derived automatically from the data via `_conditions`.

# Fields
- `sm` — surrogate model to fit
- `data` — CM summary statistics
- `prior` — SM parameter prior (bounds + names)
- `loss` — objective function (default: `GaussianNLL()`)

# Example
```julia
problem = SMFitProblem(sm, data, prior)                        # GaussianNLL by default
problem = SMFitProblem(sm, data, prior; loss = CustomLoss(fn))
result  = fitSurrogate(problem, P0)
uq      = SmoreBase._uq(problem, result, ProfileLikelihood())
samples = sampleSMPredictions(problem, uq)
```
"""
struct SMFitProblem
    sm::AbstractSurrogateModel
    data::AbstractCMData
    prior::ParameterPrior
    loss::AbstractLoss
end

SMFitProblem(sm, data, prior; loss::AbstractLoss = GaussianNLL()) =
    SMFitProblem(sm, data, prior, loss)
