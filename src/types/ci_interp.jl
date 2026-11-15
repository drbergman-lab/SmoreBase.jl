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
    CIBoundsInterpolant{I} <: Function

Callable returned by `_buildBoundsInterpolant` for a `GridCMSample`:
`get_bounds(θ_CM::AbstractVector) -> (lb::Vector{Float64}, ub::Vector{Float64})`, the SM
parameter CI bounds interpolated at CM point `θ_CM`.

Holds one interpolant per SM parameter per bound (`itp_lb`, `itp_ub`). As a named struct, a
saved result holding it (e.g. written with JLD2) refers to it by a stable type name.
"""
struct CIBoundsInterpolant{I} <: Function
    itp_lb::Vector{I}
    itp_ub::Vector{I}
end

function (f::CIBoundsInterpolant)(θ_CM::AbstractVector)
    lb = [itp(θ_CM...) for itp in f.itp_lb]
    ub = [itp(θ_CM...) for itp in f.itp_ub]
    return lb, ub
end

Base.show(io::IO, f::CIBoundsInterpolant) =
    print(io, "CIBoundsInterpolant(", length(f.itp_lb), " SM parameters)")
Base.show(io::IO, ::MIME"text/plain", f::CIBoundsInterpolant) = show(io, f)

"""
    _buildBoundsInterpolant(layout, lb_table, ub_table, method) -> get_bounds

Return a callable `get_bounds(θ_CM::AbstractVector) -> (lb::Vector, ub::Vector)` that
interpolates SM parameter CI bounds at an arbitrary CM parameter point. For a `GridCMSample`
with `LinearCIInterp` this is a `CIBoundsInterpolant`.

`lb_table` and `ub_table` are `[n_cm_param_sets × n_sm_params]` — rows correspond to rows of
`layout.params`.
"""
function _buildBoundsInterpolant(
    layout   :: GridCMSample,
    lb_table :: Matrix{Float64},
    ub_table :: Matrix{Float64},
    ::LinearCIInterp,
)
    # One Interpolations.jl interpolant per SM parameter per bound. `reshapeToGrid` scatters
    # each per-cm_param_set-row column onto the grid.
    grid_axes = Tuple(layout.axes)
    itp(col) = interpolate(grid_axes, reshapeToGrid(layout, col), Gridded(Linear()))
    return CIBoundsInterpolant(
        [itp(lb_table[:, j]) for j in axes(lb_table, 2)],
        [itp(ub_table[:, j]) for j in axes(ub_table, 2)],
    )
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
