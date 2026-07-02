# Build the optimization objective for a single cm_param_set.
# Returns a closure `(p, _hyper) -> Float64` compatible with Optimization.jl.
function _buildObjective(
    sm::AbstractSurrogateModel,
    data_slice::AbstractCMDataSlice,
    conditions::ConditionSpec,
    loss::AbstractLoss,
)
    n_cond = length(conditions)
    times  = _times(data_slice)
    return function (p, _hyper)
        total = 0.0
        for ki in 1:n_cond
            A_pred = _evaluate(sm, times, p, conditions[ki])
            total += _computeLoss(loss, A_pred, data_slice, ki)
        end
        return total
    end
end
