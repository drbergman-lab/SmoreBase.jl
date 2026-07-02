using Interpolations: interpolate, Gridded, Linear

"""
    AbstractCIInterpolator

Abstract type for the method used to interpolate SM parameter CI bounds across CM parameter
space. Dispatch on `(AbstractCMSample, AbstractCIInterpolator)` selects the implementation.

Implemented combinations:
- `GridCMSample + LinearCIInterp` — n-linear interpolation on a regular grid

All other combinations error at `_buildBoundsInterpolant` construction time.
"""
abstract type AbstractCIInterpolator end

"""
    LinearCIInterp()

N-linear interpolation of CI bounds on a regular CM parameter grid.
Requires `GridCMSample`.
"""
struct LinearCIInterp <: AbstractCIInterpolator end

"""
    RBFCIInterp()

Radial basis function interpolation of CI bounds on scattered CM parameter points.
Requires `ScatteredCMSample`. Not yet implemented.
"""
struct RBFCIInterp <: AbstractCIInterpolator end

# ── implementations ─────────────────────────────────────────────────────────────────────

"""
    _buildBoundsInterpolant(layout, lb_table, ub_table, method) -> get_bounds

Return a callable `get_bounds(θ_CM::AbstractVector) -> (lb::Vector, ub::Vector)` that
interpolates SM parameter CI bounds at an arbitrary CM parameter point.

`lb_table` and `ub_table` are `[n_cm_param_sets × n_sm_params]` — rows correspond to rows of
`layout.params`.
"""
function _buildBoundsInterpolant(
    layout   :: GridCMSample,
    lb_table :: Matrix{Float64},
    ub_table :: Matrix{Float64},
    ::LinearCIInterp,
)
    axes = layout.axes
    n_sm = size(lb_table, 2)

    # Build one Interpolations.jl interpolant per SM parameter per bound.
    # `reshapeToGrid` scatters each per-cm_param_set-row column onto the grid.
    # Stored as Vector{Any} because the concrete interpolant type varies with n_cm.
    itp_lb = Vector{Any}(undef, n_sm)
    itp_ub = Vector{Any}(undef, n_sm)
    for j in 1:n_sm
        grid_lb = reshapeToGrid(layout, lb_table[:, j])
        grid_ub = reshapeToGrid(layout, ub_table[:, j])
        itp_lb[j] = interpolate(Tuple(axes), grid_lb, Gridded(Linear()))
        itp_ub[j] = interpolate(Tuple(axes), grid_ub, Gridded(Linear()))
    end

    return function get_bounds(θ_CM::AbstractVector)
        lb = [itp_lb[j](θ_CM...) for j in 1:n_sm]
        ub = [itp_ub[j](θ_CM...) for j in 1:n_sm]
        return lb, ub
    end
end

# Catch-all: unimplemented combination.
function _buildBoundsInterpolant(
    layout   :: AbstractCMSample,
    ::Matrix{Float64},
    ::Matrix{Float64},
    method   :: AbstractCIInterpolator,
)
    error("CI bound interpolation not implemented for $(typeof(layout)) + $(typeof(method))")
end
