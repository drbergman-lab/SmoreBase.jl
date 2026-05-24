"""
    AbstractCMSample

Abstract type representing a set of CM parameter points at which SM UQ results are known.

Concrete subtypes encode the spatial structure of the points (regular grid, scattered, etc.)
and drive dispatch in consumers such as SmoreGSA's CI-bound interpolation and SmoreFit's
grid-aware posterior.
"""
abstract type AbstractCMSample end

"""
    GridCMSample(params)

CM parameter points on a regular grid (rows = Cartesian product of per-dimension values).

# Arguments
- `params` — `[n_cohorts × n_cm_params]` matrix; rows must form the Cartesian product of
  unique sorted values in each column

# Fields
- `params` — raw matrix `[n_cohorts × n_cm_params]`
- `axes`   — `axes[d]` holds the sorted unique values along CM dimension `d`

Throws `ArgumentError` if the rows are not consistent with a Cartesian product structure.

# Example
```julia
# 1-D grid over a single CM parameter
cm_sample = GridCMSample([1.0; 2.0; 3.0; 4.0; 5.0;;])

# 2-D grid over two CM parameters (4 cohorts)
cm_sample = GridCMSample([1.0 0.1; 1.0 0.2; 2.0 0.1; 2.0 0.2])
```
"""
struct GridCMSample <: AbstractCMSample
    params :: Matrix{Float64}
    axes   :: Vector{Vector{Float64}}
end

function GridCMSample(params::AbstractMatrix)
    mat  = Matrix{Float64}(params)
    axes = [sort(unique(c)) for c in eachcol(mat)]
    n_rows = size(mat, 1)
    n_rows == prod(length.(axes)) || throw(ArgumentError(
        "GridCMSample: expected $(prod(length.(axes))) rows (Cartesian product of " *
        "per-dimension unique values) but got $n_rows"
    ))
    length(Set(Tuple(r) for r in eachrow(mat))) == n_rows || throw(ArgumentError(
        "GridCMSample: rows are not unique — not a valid Cartesian product grid"
    ))
    return GridCMSample(mat, axes)
end

"""
    ScatteredCMSample(params)

CM parameter points at arbitrary (non-grid) locations.

# Arguments
- `params` — `[n_cohorts × n_cm_params]` matrix

# Note
Interpolation support for scattered layouts is not yet implemented. When implementing,
consider adding a `kdtree::KDTree` field (from `NearestNeighbors.jl`) built in the
constructor to avoid rebuilding per call for local methods (IDW, local RBF with k-nearest).
"""
struct ScatteredCMSample <: AbstractCMSample
    params :: Matrix{Float64}
end

ScatteredCMSample(params::AbstractMatrix) = ScatteredCMSample(Matrix{Float64}(params))

"""
    CMSample(params::AbstractMatrix) -> AbstractCMSample

Build a `GridCMSample` from `params` if its rows form a regular Cartesian-product grid,
otherwise fall back to a `ScatteredCMSample`. This is the convenience factory consumers use
to accept a raw `[n_cohorts × n_cm_params]` matrix without committing to a layout up front.

A fallback to `ScatteredCMSample` is reported via `@info`, since it usually indicates the
caller expected a grid but the values do not line up (often floating-point inconsistency).

# Example
```julia
CMSample([1.0 0.1; 1.0 0.2; 2.0 0.1; 2.0 0.2])  # → GridCMSample (2×2 grid)
CMSample([0.13 0.7; 0.42 0.2; 0.91 0.55])        # → ScatteredCMSample
```
"""
function CMSample(params::AbstractMatrix)
    try
        return GridCMSample(params)
    catch e
        e isa ArgumentError || rethrow()
        @info "CM parameter matrix could not be interpreted as a regular grid; " *
              "falling back to ScatteredCMSample. If you expected a grid layout, " *
              "check for floating-point inconsistencies in your CM parameter values."
        return ScatteredCMSample(params)
    end
end

"""
    _gridIndices(g::GridCMSample) -> Vector{CartesianIndex}

Map each cohort row of `g.params` to its `CartesianIndex` in the grid spanned by `g.axes`.
The `k`-th entry locates row `k` so that per-row vectors can be scattered onto a grid array.
"""
function _gridIndices(g::GridCMSample)
    axes   = g.axes
    params = g.params
    n_cm   = length(axes)
    return [
        CartesianIndex(ntuple(d -> searchsortedfirst(axes[d], params[k, d]), n_cm))
        for k in 1:size(params, 1)
    ]
end

"""
    reshapeToGrid(g::GridCMSample, v::AbstractVector) -> Array

Reshape a per-cohort-row vector `v` (one value per row of `g.params`) onto the CM grid, an
array of size `length.(g.axes)`. Entry positions follow `_gridIndices(g)`, so `v[k]` lands at
the grid cell of cohort row `k`.

Only defined for `GridCMSample`; scattered layouts have no grid to reshape onto.

# Example
```julia
g = GridCMSample([1.0 0.1; 1.0 0.2; 2.0 0.1; 2.0 0.2])
reshapeToGrid(g, [11, 12, 21, 22])   # → 2×2 Matrix
```
"""
function reshapeToGrid(g::GridCMSample, v::AbstractVector)
    length(v) == size(g.params, 1) || throw(ArgumentError(
        "reshapeToGrid: length(v)=$(length(v)) must equal the number of cohort points " *
        "$(size(g.params, 1))"
    ))
    idx = _gridIndices(g)
    out = Array{eltype(v)}(undef, length.(g.axes)...)
    for k in eachindex(v)
        out[idx[k]] = v[k]
    end
    return out
end

reshapeToGrid(s::AbstractCMSample, ::AbstractVector) =
    error("reshapeToGrid is only defined for GridCMSample, got $(typeof(s))")
