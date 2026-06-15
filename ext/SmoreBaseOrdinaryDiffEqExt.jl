module SmoreBaseOrdinaryDiffEqExt

using SmoreBase
using OrdinaryDiffEq

# Override the stub that errors without the extension loaded.
function SmoreBase._evaluate(sm::SmoreBase.ODESurrogateModel, t, p, condition)
    # Apply the preprocessor first so both the custom-solve and default-solve
    # branches operate on the same (preprocessed) inputs.
    p, condition = SmoreBase._applyPreprocessor(sm, p, condition)
    y0 = sm.y0

    result = if sm.custom_solve_fn !== nothing
        sm.custom_solve_fn(sm, t, p, condition, y0)
    else
        tspan = (Float64(t[1]), Float64(t[end]))
        prob  = ODEProblem(sm.ode_fn, y0, tspan, p)
        sol   = solve(prob, sm.solver;
                      abstol  = sm.abstol,
                      reltol  = sm.reltol,
                      saveat  = t)

        # Build [n_times × n_outputs] matrix
        state_mat = reduce(hcat, sol.u)'  # [n_times × n_states]

        if sm.output_variables !== nothing
            state_mat[:, sm.output_variables]
        else
            state_mat
        end
    end

    return SmoreBase._applyPostprocessor(sm, result)
end

end # module
