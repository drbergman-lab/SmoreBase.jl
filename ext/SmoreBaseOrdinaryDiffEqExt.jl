module SmoreBaseOrdinaryDiffEqExt

using SmoreBase
using OrdinaryDiffEq

# Override the stub that errors without the extension loaded.
function SmoreBase._evaluate(sm::SmoreBase.ODESurrogateModel, t, p, condition)
    p, condition = SmoreBase._applyPreprocessor(sm, p, condition)
    y0 = sm.y0

    # tspan starts at sm.t0 (the time at which y0 holds), not t[1] — observations are
    # requested at t via saveat, independent of where the solve itself starts.
    tspan = (sm.t0, Float64(t[end]))
    prob  = ODEProblem(sm.ode_fn, y0, tspan, p)
    sol   = solve(prob, sm.solver;
                  abstol  = sm.abstol,
                  reltol  = sm.reltol,
                  saveat  = t)

    # Build [n_times × n_outputs] matrix
    state_mat = reduce(hcat, sol.u)'  # [n_times × n_states]

    result = if sm.output_variables !== nothing
        state_mat[:, sm.output_variables]
    else
        state_mat
    end

    return SmoreBase._applyPostprocessor(sm, result)
end

end # module
