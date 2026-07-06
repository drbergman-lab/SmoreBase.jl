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
    CustomSurrogateModel(; fn, kwargs...)

Surrogate model defined by an arbitrary user-supplied function.

`fn` may be a closed-form analytical solution, a numerical solve (e.g. a PDE method-of-lines
integration), a lookup table, or any other mapping. Anything the function needs beyond
`(t, p, condition)` — such as an initial condition, a spatial mesh, or solver settings — is
captured in its closure.

`fn` signature: `(t::Vector, p::Vector, condition) -> Matrix{Float64}`
where rows are time points and columns are output variables. Receives the **preprocessed**
`(p, condition)`.

# Fields
- `fn` — surrogate evaluation function
- `pre_processor` — `Union{Nothing,Function}` of the form `(p, condition) -> (p_new, condition_new)`, applied before evaluation
- `post_processor` — `Union{Nothing,Function}` applied to the prediction matrix after evaluation

# Examples
```julia
# Closed-form logistic solution.
sm = CustomSurrogateModel(
    fn = (t, p, c) -> reshape(p[2] ./ (1 .+ (p[2]/0.01 - 1) .* exp.(-p[1] .* t)), :, 1),
)

# Numerical solve closing over an initial condition.
y0 = [0.01]
sm = CustomSurrogateModel(
    fn = (t, p, _c) -> reshape(y0[1] .+ p[1] .* t, :, 1),
)
```
"""
struct CustomSurrogateModel{F,Pre,Post} <: AbstractSurrogateModel
    fn::F
    pre_processor::Pre
    post_processor::Post
end

function CustomSurrogateModel(;
    fn,
    pre_processor   = nothing,
    post_processor  = nothing,
)
    return CustomSurrogateModel(fn, pre_processor, post_processor)
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

For both built-in surrogate model types (`CustomSurrogateModel` and `ODESurrogateModel`), this
applies the `pre_processor` to `(p, condition)`, evaluates the model, then applies the
`post_processor` to the resulting prediction matrix.
"""
function _evaluate(sm::CustomSurrogateModel, t, p, condition)
    p_eff, c_eff = _applyPreprocessor(sm, p, condition)
    result = sm.fn(t, p_eff, c_eff)
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
