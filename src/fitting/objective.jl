# Build the optimization objective for a single param_set.
# Returns a closure `(p, _hyper) -> Float64` compatible with Optimization.jl.
function _buildObjective(
    sm::AbstractSurrogateModel,
    data::AbstractCMData,
    conditions::ConditionSpec,
    loss::AbstractLoss,
    param_set_idx::Int,
)
    n_cond = length(conditions)
    times  = data.times
    return function (p, _hyper)
        total = 0.0
        for ki in 1:n_cond
            A_pred = _evaluate(sm, times, p, conditions[ki])
            total += _computeLoss(loss, A_pred, data, param_set_idx, ki)
        end
        return total
    end
end
