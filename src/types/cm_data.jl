"""
    AbstractCMData

Abstract base type for CM observation containers.
"""
abstract type AbstractCMData end

"""
    AbstractCMDataSlice <: AbstractCMData

Abstract base type for a single-param-set view into an `AbstractCMData` container.
A slice has no param-set axis; `n_param_sets` is not defined on this type.

Custom `AbstractCMData` subtypes must implement:
```julia
_sliceParamSet(data::MyType, pi::Int) -> AbstractCMDataSlice  # always required
n_param_sets(data::MyType) -> Int                             # required by fitSurrogate
```
The following have working defaults but should be overridden when applicable:
```julia
_conditions(data::MyType) -> ConditionSpec   # default: ConditionSpec() (single condition)
_times(data::MyType) -> Union{Nothing,Vector} # default: nothing (endpoint data)
```
`n_conditions` and `n_times` are derived automatically from `_conditions` and `_times`
respectively — do not implement them unless you need a more efficient specialization.

For `AbstractCMDataSlice` subtypes (needed only if not returning `CMDataSlice` from
`_sliceParamSet`):
```julia
_times(slice::MySlice) -> Union{Nothing,Vector}  # default: nothing; override for time-series
_mean(slice::MySlice) -> AbstractArray           # required if using GaussianNLL
_sd(slice::MySlice)   -> AbstractArray           # required if using GaussianNLL
_cov(slice::MySlice)  -> Union{Nothing,AbstractArray}  # required if using GaussianNLL
```
"""
abstract type AbstractCMDataSlice <: AbstractCMData end

"""
    CMData{T<:Real}

Structured container for summary statistics from CM simulation runs used to train a surrogate model.

Arrays are stored in canonical 4-D layout: `[n_times, n_variables, n_conditions, n_param_sets]`.

The axis order reflects the natural data hierarchy: a single CM run at one parameter set and
one condition produces a `[n_times × n_variables]` matrix; conditions and parameter sets are
the outer two axes.

# Fields
- `μ` — mean observations `[n_times, n_variables, n_conditions, n_param_sets]`
- `σ` — pointwise noise level treated as a known standard deviation in the likelihood (same shape as `μ`);
  the caller controls the convention — pass the sample standard deviation `σ_rep` for a fixed noise
  floor, or the standard error `σ_rep/√R` if the likelihood should tighten with replicate count `R`
- `Σ` — optional full covariance `[n_variables, n_variables, n_times, n_conditions, n_param_sets]` (`nothing` → independent)
- `times` — time grid (`nothing` when the time axis is absent)
- `variable_labels` — names of observable output variables
- `condition_labels` — labels for experimental conditions
- `param_set_labels` — labels for CM parameter vectors

# Keyword constructor

All four axes are controlled by kwargs that each accept:
- `nothing` (default) — axis is absent from the lower-D input; size is 1
- `Int` — axis is present; gives the size; labels are auto-generated (`times` excluded)
- `Vector` — axis is present; gives the labels and size (`times` stores actual time values)

For an N-D input array, exactly N of the four axis kwargs must be non-nothing.
The N kwargs identify which axis occupies each dimension of the input array.

**Auto-matching:** when all N kwarg sizes are distinct, each array dimension is matched to the
unique kwarg whose size equals `size(A, dim)`, and the array is permuted into canonical order.

**`dim_order`:** required when any two present kwargs have the same size (ambiguous matching).
Supply a length-N `Vector` or `Tuple` of `Symbol`s (`:times`, `:variables`, `:conditions`,
`:param_sets`) declaring which axis occupies each array dimension. The array is then validated
and permuted into canonical order.

Both Unicode (`μ`, `σ`, `Σ`) and ASCII (`mean`, `sd`, `cov`) aliases are accepted.
Supplying both forms for the same field throws `ArgumentError`.

# Examples
```julia
t = collect(0.0:1.0:5.0)

# 1-D: single variable, condition, param-set
data = CMData(mean = rand(6), sd = 0.1 .* ones(6), times = t)

# 2-D [n_times × n_variables]: two outputs
data = CMData(μ = rand(6, 2), σ = 0.1 .* ones(6, 2), times = t,
              variables = ["tumor", "immune"])

# 2-D [n_times × n_conditions]: auto-matched even in non-canonical order
data = CMData(μ = rand(3, 6), σ = 0.1 .* ones(3, 6),
              times = t, conditions = 3)

# 2-D with repeated sizes: dim_order required to resolve ambiguity
data = CMData(μ = rand(6, 6), σ = 0.1 .* ones(6, 6),
              times = t, conditions = 6,
              dim_order = [:times, :conditions])

# No time axis: endpoint data [n_variables × n_conditions]
data = CMData(μ = rand(2, 3), σ = 0.1 .* ones(2, 3),
              variables = ["tumor", "immune"], conditions = 3)
```
"""
struct CMData{T<:Real} <: AbstractCMData
    μ::Array{T,4}                  # [n_times, n_variables, n_conditions, n_param_sets]
    σ::Array{T,4}
    Σ::Union{Nothing,Array{T}}     # [n_variables, n_variables, n_times, n_conditions, n_param_sets]
    times::Union{Nothing,Vector{T}}
    variable_labels::Vector{String}
    condition_labels::Vector{String}
    param_set_labels::Vector{String}
end

function CMData(;
    μ = nothing, mean = nothing,
    σ = nothing, sd   = nothing,
    Σ = nothing, cov  = nothing,
    times      = nothing,
    variables  = nothing,
    conditions = nothing,
    param_sets = nothing,
    dim_order  = nothing,
)
    _μ = _resolveRequiredAlias(μ, mean, "μ", "mean")
    _σ = _resolveRequiredAlias(σ, sd,   "σ", "sd")
    _Σ = _resolveOptionalAlias(Σ, cov,  "Σ", "cov")

    n_t  = _axisSize(times)
    n_v  = _axisSize(variables)
    n_c  = _axisSize(conditions)
    n_ps = _axisSize(param_sets)

    _μ4 = _reshapeTo4D(_μ, n_t, n_v, n_c, n_ps, times, variables, conditions, param_sets, dim_order)
    _σ4 = _reshapeTo4D(_σ, n_t, n_v, n_c, n_ps, times, variables, conditions, param_sets, dim_order)

    size(_μ4) == size(_σ4) ||
        throw(ArgumentError("μ and σ must have the same shape; got $(size(_μ4)) vs $(size(_σ4))"))

    _var_labels  = _axisLabels(variables,  n_v,  "y")
    _cond_labels = _axisLabels(conditions, n_c,  "c")
    _ps_labels   = _axisLabels(param_sets, n_ps, "ps")

    !isnothing(_Σ) && _validateCovariance(_Σ, size(_μ4, 1), size(_μ4, 2), size(_μ4, 3), size(_μ4, 4))

    T = float(eltype(_μ4))
    return CMData{T}(
        convert(Array{T,4}, _μ4),
        convert(Array{T,4}, _σ4),
        isnothing(_Σ) ? nothing : convert(Array{T}, _Σ),
        isnothing(times) ? nothing : convert(Vector{T}, times),
        _var_labels,
        _cond_labels,
        _ps_labels,
    )
end

# ── private helpers ───────────────────────────────────────────────────────────

function _resolveRequiredAlias(unicode_val, ascii_val, unicode_name, ascii_name)
    !isnothing(unicode_val) && !isnothing(ascii_val) &&
        throw(ArgumentError("Supply either `$unicode_name` or `$ascii_name`, not both"))
    val = !isnothing(unicode_val) ? unicode_val : ascii_val
    isnothing(val) && throw(ArgumentError("Must supply `$unicode_name` (or `$ascii_name`)"))
    return val
end

function _resolveOptionalAlias(unicode_val, ascii_val, unicode_name, ascii_name)
    !isnothing(unicode_val) && !isnothing(ascii_val) &&
        throw(ArgumentError("Supply either `$unicode_name` or `$ascii_name`, not both"))
    return !isnothing(unicode_val) ? unicode_val : ascii_val
end

_axisSize(::Nothing)         = 1
_axisSize(n::Integer)        = Int(n)
_axisSize(v::AbstractVector) = length(v)

function _axisLabels(kwarg, n::Int, prefix::String)
    kwarg isa AbstractVector && return String.(kwarg)
    return n == 1 ? ["$(prefix)1"] : ["$(prefix)$(i)" for i in 1:n]
end

# Reshape A to canonical 4-D [n_t, n_v, n_c, n_ps].
#
# Exactly ndims(A) of the four axis kwargs must be non-nothing. Each non-nothing kwarg
# identifies one axis. The N present axes are matched to the N array dimensions either
# automatically (by unique size) or via the explicit dim_order argument. The result is
# permuted into canonical order and reshaped to 4-D.
function _reshapeTo4D(A::AbstractArray, n_t, n_v, n_c, n_ps,
                      times_kwarg, variables_kwarg, conditions_kwarg, param_sets_kwarg,
                      dim_order)
    full_shape = (n_t, n_v, n_c, n_ps)
    N = ndims(A)

    # Collect present axes in canonical order
    present_syms  = Symbol[]
    present_sizes = Int[]
    if !isnothing(times_kwarg); push!(present_syms, :times);      push!(present_sizes, n_t);  end
    if !isnothing(variables_kwarg); push!(present_syms, :variables);  push!(present_sizes, n_v);  end
    if !isnothing(conditions_kwarg); push!(present_syms, :conditions); push!(present_sizes, n_c);  end
    if !isnothing(param_sets_kwarg); push!(present_syms, :param_sets); push!(present_sizes, n_ps); end
    n_present = length(present_syms)

    n_present == N ||
        throw(ArgumentError(
            "$(N)-D μ/σ requires exactly $N non-nothing axis kwargs " *
            "(got $n_present: [$(join(present_syms, ", "))])"
        ))

    present_dict = Dict(zip(present_syms, present_sizes))

    # Determine which axis occupies each input dimension
    dim_axes = if !isnothing(dim_order)
        _dimAxesFromOrder(A, [Symbol(s) for s in dim_order], present_dict)
    elseif length(unique(present_sizes)) < n_present
        dup = unique(s for s in present_sizes if count(==(s), present_sizes) > 1)
        throw(ArgumentError(
            "Axis sizes $dup appear more than once among the present kwargs " *
            "$(present_syms); supply `dim_order` to resolve the ambiguity.\n" *
            "  Example: dim_order = $(present_syms)"
        ))
    else
        _dimAxesFromSizes(A, present_dict)
    end

    # Build permutation to canonical order [:times, :variables, :conditions, :param_sets]
    canonical = [:times, :variables, :conditions, :param_sets]
    perm = [findfirst(==(sym), dim_axes)
            for sym in canonical if haskey(present_dict, sym)]

    A_ordered = perm == collect(1:N) ? A : permutedims(A, perm)
    return reshape(A_ordered, full_shape)
end

# Match array dims to axes by unique size.
function _dimAxesFromSizes(A::AbstractArray, present_dict::Dict{Symbol,Int})
    size_to_sym = Dict(v => k for (k, v) in present_dict)
    dim_axes = Vector{Symbol}(undef, ndims(A))
    for i in 1:ndims(A)
        s = size(A, i)
        haskey(size_to_sym, s) ||
            throw(ArgumentError(
                "dimension $i has size $s which does not match any present axis size " *
                "(present: $(join(["$k → size $v" for (k, v) in present_dict], ", ")))"
            ))
        dim_axes[i] = size_to_sym[s]
    end
    # Verify bijection: every present axis matched exactly once
    for sym in keys(present_dict)
        sym in dim_axes ||
            throw(ArgumentError(
                "axis :$sym (size $(present_dict[sym])) was not matched to any array " *
                "dimension; check that the array shape $(size(A)) is consistent with " *
                "the axis sizes"
            ))
    end
    return dim_axes
end

# Match array dims to axes via explicit dim_order.
function _dimAxesFromOrder(A::AbstractArray, dim_syms::Vector{Symbol},
                           present_dict::Dict{Symbol,Int})
    N = ndims(A)
    length(dim_syms) == N ||
        throw(ArgumentError(
            "dim_order has $(length(dim_syms)) entries but array has $N dimensions"
        ))
    length(unique(dim_syms)) == N ||
        throw(ArgumentError("dim_order has repeated axis names; each axis must appear once"))
    for (i, sym) in enumerate(dim_syms)
        haskey(present_dict, sym) ||
            throw(ArgumentError(
                "dim_order refers to :$sym which is not among the non-nothing axis kwargs " *
                "(present: $(collect(keys(present_dict))))"
            ))
        size(A, i) == present_dict[sym] ||
            throw(ArgumentError(
                "dim_order says dimension $i is :$sym (expected size $(present_dict[sym])) " *
                "but the array has size $(size(A, i)) in that dimension"
            ))
    end
    return dim_syms
end

function _validateCovariance(Σ, n_t, n_v, n_c, n_ps)
    ndims(Σ) == 5 ||
        throw(ArgumentError(
            "Σ must be a 5-D array [n_variables, n_variables, n_times, n_conditions, n_param_sets]; " *
            "got $(ndims(Σ))-D"
        ))
    size(Σ) == (n_v, n_v, n_t, n_c, n_ps) ||
        throw(ArgumentError("Σ shape must be ($n_v, $n_v, $n_t, $n_c, $n_ps); got $(size(Σ))"))
end

# ── convenience accessors ─────────────────────────────────────────────────────

n_times(d::CMData)      = size(d.μ, 1)
n_variables(d::CMData)  = size(d.μ, 2)
n_conditions(d::CMData) = size(d.μ, 3)
n_param_sets(d::CMData) = size(d.μ, 4)

# Defaults for AbstractCMData: derived from other required accessors.
# Custom subtypes get these for free if they implement _times and _conditions.
function n_times(d::AbstractCMData)
    times = _times(d)
    return isnothing(times) ? 1 : length(times)
end
n_conditions(d::AbstractCMData) = length(_conditions(d))

"""
    _times(d::AbstractCMData) -> Union{Nothing, Vector}

Return the time grid for `d`, or `nothing` if the time axis is absent.
Default implementation returns `nothing`; `CMData` overrides with its `times` field.
Custom subtypes may override to expose their own time grid.
"""
_times(::AbstractCMData)          = nothing
_times(d::CMData)                 = d.times

"""
    _variableLabels(d::AbstractCMData) -> Union{Nothing, Vector{String}}

Return the variable labels for `d`, or `nothing` if unlabelled.
Default returns `nothing`; `CMData` overrides with its `variable_labels` field.
"""
_variableLabels(::AbstractCMData) = nothing
_variableLabels(d::CMData)        = d.variable_labels

"""
    _conditionLabels(d::AbstractCMData) -> Union{Nothing, Vector{String}}

Return the condition labels for `d`, or `nothing` if unlabelled.
Default returns `nothing`; `CMData` overrides with its `condition_labels` field.
"""
_conditionLabels(::AbstractCMData) = nothing
_conditionLabels(d::CMData)        = d.condition_labels

"""
    _conditions(d::AbstractCMData) -> ConditionSpec

Return the experimental conditions for `d` as a `ConditionSpec`.
The default returns `ConditionSpec()` (single `"default"` condition).
Custom `AbstractCMData` subtypes with multiple conditions must override this method.
`CMData` overrides with its `condition_labels` field.
"""
_conditions(::AbstractCMData) = ConditionSpec()
_conditions(d::CMData)        = ConditionSpec(d.condition_labels)

"""
    _paramSetLabels(d::AbstractCMData) -> Union{Nothing, Vector{String}}

Return the param-set labels for `d`, or `nothing` if unlabelled.
Default returns `nothing`; `CMData` overrides with its `param_set_labels` field.
"""
_paramSetLabels(::AbstractCMData) = nothing
_paramSetLabels(d::CMData)        = d.param_set_labels

# ── CMDataSlice ───────────────────────────────────────────────────────────────

"""
    CMDataSlice{T<:Real} <: AbstractCMDataSlice

A zero-copy view into a single param-set of a `CMData` container. Fields `μ`, `σ`,
and (optionally) `Σ` are `SubArray` views into the parent arrays — no data is copied.

Created by `_sliceParamSet(data::CMData, pi)`.

# Fields
- `μ` — mean view `[n_times, n_variables, n_conditions]`
- `σ` — noise view (same shape as `μ`)
- `Σ` — optional full-covariance view `[n_variables, n_variables, n_times, n_conditions]`, or `nothing`
- `times` — shared reference to the parent time grid, or `nothing`
- `variable_labels`, `condition_labels` — shared references to parent label vectors, or `nothing`
- `param_set_label` — label string for this param-set, or `nothing`
"""
struct CMDataSlice{T<:Real} <: AbstractCMDataSlice
    μ::AbstractArray{T,3}                       # [n_times, n_variables, n_conditions]
    σ::AbstractArray{T,3}
    Σ::Union{Nothing,AbstractArray{T,4}}        # [n_variables, n_variables, n_times, n_conditions]
    times::Union{Nothing,Vector{T}}
    variable_labels::Union{Nothing,Vector{String}}
    condition_labels::Union{Nothing,Vector{String}}
    param_set_label::Union{Nothing,String}
end

"""
    _sliceParamSet(data::CMData, pi::Int) -> CMDataSlice

Return a zero-copy view of `data` restricted to param-set index `pi`.

Custom `AbstractCMData` subtypes must implement their own method returning an
`AbstractCMDataSlice`. The slice is passed to `_computeLoss` and to user-supplied
`CustomLoss` functions as the `data` argument.
"""
function _sliceParamSet(data::CMData, pi::Int)
    return CMDataSlice(
        @view(data.μ[:, :, :, pi]),
        @view(data.σ[:, :, :, pi]),
        isnothing(data.Σ) ? nothing : @view(data.Σ[:, :, :, :, pi]),
        _times(data),
        _variableLabels(data),
        _conditionLabels(data),
        _paramSetLabels(data)[pi],
    )
end

n_times(d::CMDataSlice)      = size(d.μ, 1)
n_variables(d::CMDataSlice)  = size(d.μ, 2)
n_conditions(d::CMDataSlice) = size(d.μ, 3)

_times(d::CMDataSlice)           = d.times
_variableLabels(d::CMDataSlice)  = d.variable_labels
_conditionLabels(d::CMDataSlice) = d.condition_labels

"""
    _mean(d::AbstractCMDataSlice) -> AbstractArray

Return the mean array for this slice, shape `[n_times, n_variables, n_conditions]`.

Custom `AbstractCMDataSlice` subtypes must implement this method to use `GaussianNLL`.
Field names may use any convention — Unicode is not required.
"""
_mean(d::CMDataSlice) = d.μ

"""
    _sd(d::AbstractCMDataSlice) -> AbstractArray

Return the standard-deviation array for this slice, same shape as `_mean`.

Custom `AbstractCMDataSlice` subtypes must implement this method to use `GaussianNLL`.
"""
_sd(d::CMDataSlice) = d.σ

"""
    _cov(d::AbstractCMDataSlice) -> Union{Nothing, AbstractArray}

Return the full covariance array for this slice
(`[n_variables, n_variables, n_times, n_conditions]`), or `nothing` for independent noise.

Custom `AbstractCMDataSlice` subtypes must implement this method to use `GaussianNLL`.
"""
_cov(d::CMDataSlice) = d.Σ
