"""
    AbstractSurrogateModel

Abstract base type for surrogate models.
"""
abstract type AbstractSurrogateModel end

"""
    ODESurrogateModel(; ode_fn, y0, solver, kwargs...)

Surrogate model backed by an ODE system (SciML convention).

Calling `_evaluate` on this type requires `using OrdinaryDiffEq` to activate
the package extension `SmoreBaseOrdinaryDiffEqExt`.

# Fields
- `ode_fn` — in-place ODE RHS: `f!(du, u, p, t)`
- `t0` — start of the ODE solve `tspan`; the state at `t0` is `y0`. Observations are returned at
  the requested `t` via `saveat`, independent of `t0` (default `0.0`)
- `y0` — initial conditions (`Vector{Float64}`)
- `solver` — ODE algorithm (e.g., `Tsit5()`)
- `output_variables` — indices of state variables that correspond to observables (`nothing` → all)
- `pre_processor` — `Union{Nothing,Function}` of the form `(p, condition) -> (p_new, condition_new)`, applied before solving
- `post_processor` — `Union{Nothing,Function}` applied to the prediction matrix after solving
- `abstol`, `reltol` — ODE solver tolerances

# Example
```julia
using OrdinaryDiffEq
sm = ODESurrogateModel(
    ode_fn = (du, u, p, t) -> (du[1] = p[1] * u[1] * (1 - u[1] / p[2])),
    y0     = [0.01],
    solver = Tsit5(),
)
```
"""
struct ODESurrogateModel{F,Pre,Post} <: AbstractSurrogateModel
    ode_fn::F
    t0::Float64
    y0::Vector{Float64}
    solver::Any
    output_variables::Union{Nothing,Vector{Int}}
    pre_processor::Pre
    post_processor::Post
    abstol::Float64
    reltol::Float64
end

function ODESurrogateModel(;
    ode_fn,
    y0,
    solver,
    output_variables = nothing,
    pre_processor    = nothing,
    post_processor   = nothing,
    t0::Real         = 0.0,
    abstol::Real     = 1e-6,
    reltol::Real     = 1e-3,
)
    return ODESurrogateModel(
        ode_fn, Float64(t0), y0, solver,
        output_variables,
        pre_processor, post_processor,
        Float64(abstol), Float64(reltol),
    )
end

"""
    AnalyticalSurrogateModel(; fn, kwargs...)

Surrogate model defined by a closed-form analytical function.

`fn` signature: `(t::Vector, p::Vector, condition::String) -> Matrix{Float64}`
where rows are time points and columns are output variables.

# Fields
- `fn` — analytical solution function
- `pre_processor` — `Union{Nothing,Function}` of the form `(p, condition) -> (p_new, condition_new)`, applied before evaluation
- `post_processor` — `Union{Nothing,Function}` applied to the prediction matrix after evaluation

# Example
```julia
sm = AnalyticalSurrogateModel(
    fn = (t, p, c) -> reshape(p[2] ./ (1 .+ (p[2]/0.01 - 1) .* exp.(-p[1] .* t)), :, 1),
)
```
"""
struct AnalyticalSurrogateModel{F,Pre,Post} <: AbstractSurrogateModel
    fn::F
    pre_processor::Pre
    post_processor::Post
end

function AnalyticalSurrogateModel(;
    fn,
    pre_processor   = nothing,
    post_processor  = nothing,
)
    return AnalyticalSurrogateModel(fn, pre_processor, post_processor)
end

"""
    CustomSolverSurrogateModel(; solve_fn, y0, kwargs...)

Surrogate model whose trajectory is produced by a user-supplied solver, entirely bypassing
`OrdinaryDiffEq` (no package extension needed — `_evaluate` is defined here in the main package).

`solve_fn` signature: `(t::Vector, p::Vector, condition, y0::Vector{Float64}) -> Matrix{Float64}`
where rows are time points and columns are output variables. Receives the **preprocessed**
`(p, condition)`.

# Fields
- `solve_fn` — custom solve function
- `y0` — initial conditions (`Vector{Float64}`), passed through to `solve_fn`
- `pre_processor` — `Union{Nothing,Function}` of the form `(p, condition) -> (p_new, condition_new)`, applied before solving
- `post_processor` — `Union{Nothing,Function}` applied to the prediction matrix after solving

# Example
```julia
sm = CustomSolverSurrogateModel(
    solve_fn = (t, p, _c, y0) -> reshape(y0[1] .+ p[1] .* t, :, 1),
    y0       = [0.01],
)
```
"""
struct CustomSolverSurrogateModel{F,Pre,Post} <: AbstractSurrogateModel
    solve_fn::F
    y0::Vector{Float64}
    pre_processor::Pre
    post_processor::Post
end

function CustomSolverSurrogateModel(;
    solve_fn,
    y0,
    pre_processor   = nothing,
    post_processor  = nothing,
)
    return CustomSolverSurrogateModel(solve_fn, y0, pre_processor, post_processor)
end

# ── internal evaluation helpers ───────────────────────────────────────────────

# pre_processor signature: (p, condition) -> (p_new, condition_new)
# Common uses: log-space → linear parameter transform, condition-dependent parameter adjustments.
function _applyPreprocessor(sm, p, condition)
    isnothing(sm.pre_processor) && return (p, condition)
    return sm.pre_processor(p, condition)
end

function _applyPostprocessor(sm, result)
    isnothing(sm.post_processor) && return result
    return sm.post_processor(result)
end

"""
    _evaluate(sm, t, p, condition) -> Matrix{Float64}

Internal evaluation entry point. Returns a `[n_times × n_outputs]` matrix of SM predictions.

For `AnalyticalSurrogateModel`: calls `sm.fn(t, p, condition)`.
For `ODESurrogateModel`: requires the `OrdinaryDiffEq` extension to be loaded.
"""
function _evaluate(sm::AnalyticalSurrogateModel, t, p, condition)
    p_eff, c_eff = _applyPreprocessor(sm, p, condition)
    result = sm.fn(t, p_eff, c_eff)
    return _applyPostprocessor(sm, result)
end

function _evaluate(sm::CustomSolverSurrogateModel, t, p, condition)
    p_eff, c_eff = _applyPreprocessor(sm, p, condition)
    result = sm.solve_fn(t, p_eff, c_eff, sm.y0)
    return _applyPostprocessor(sm, result)
end

# Generic fallback: catches any AbstractSurrogateModel subtype that has no _evaluate method.
# The ODE extension overrides this for ODESurrogateModel specifically.
function _evaluate(sm::AbstractSurrogateModel, args...)
    msg = "No `_evaluate` method defined for $(typeof(sm))."
    if sm isa ODESurrogateModel
        msg *= " Load OrdinaryDiffEq first (`using OrdinaryDiffEq`) to activate the ODE extension."
    end
    error(msg)
end
