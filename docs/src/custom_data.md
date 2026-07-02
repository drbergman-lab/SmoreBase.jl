# Custom CM Data Types

SmoreBase ships with `CMData`, a general-purpose container for complex-model (CM) summary statistics. For cases where `CMData` does not fit — e.g. your CM produces structured outputs that don't map cleanly onto a `[n_times × n_variables × n_conditions × n_cm_param_sets]` array — you can subtype `AbstractCMData` and teach SmoreBase how to use your type.

## Required interface

### On your `AbstractCMData` subtype

Always implement:

- `_sliceCmParamSet(data::MyType, pi::Int) -> AbstractCMDataSlice`
- `n_cm_param_sets(data::MyType) -> Int` — `fitSurrogate` uses this to validate `P0`

Override the defaults when applicable (both default to single-condition / no-time-axis behaviour):

- `_conditions(data::MyType) -> ConditionSpec` — default: `ConditionSpec()` (one `"default"` condition); override if your data has multiple experimental conditions
- `_times(data::MyType) -> Union{Nothing,Vector}` — default: `nothing`; override if your data has a time axis

**Derived for free — do not implement:**

- `n_conditions` — automatically `length(_conditions(data))`
- `n_times` — automatically `1` when `_times` returns `nothing`, or `length(_times(data))` otherwise
- `n_variables` — not used by the pipeline; implement only if you need it for your own inspection code

### On your `AbstractCMDataSlice` subtype

Only needed if you define a custom slice type rather than returning `CMDataSlice` from `_sliceCmParamSet`:

- `_times(slice::MySlice)` — default: `nothing`; override for time-series data (the SM is evaluated at these times)
- `_mean`, `_sd`, `_cov` — required if using `GaussianNLL`

### Optional

- `CustomLoss` — for fully custom loss logic that doesn't use `GaussianNLL`

### 1. `_sliceCmParamSet`

```julia
SmoreBase._sliceCmParamSet(data::MyData, pi::Int) -> AbstractCMDataSlice
```

Return an `AbstractCMDataSlice` containing only the data for param-set `pi`. Some guidelines:

- **Array slices** (sub-arrays indexed by `pi`): use `@view` to avoid copying — each param-set gets its own slice, so copying would duplicate your entire dataset once per param-set.
- **Whole arrays** (like `times` or label vectors): pass directly. Julia passes arrays by reference, so no copy occurs.

The simplest option is to return a `CMDataSlice` built from views into your own arrays:

```julia
function SmoreBase._sliceCmParamSet(data::MyData, pi::Int)
    ps_labels = SmoreBase._cmParamSetLabels(data)
    return CMDataSlice(
        @view(data.mean_arr[:, :, :, pi]),
        @view(data.sd_arr[:, :, :, pi]),
        nothing,                                              # no full covariance
        SmoreBase._times(data),                               # nothing if absent
        SmoreBase._variableLabels(data),                      # nothing if unlabelled
        SmoreBase._conditionLabels(data),                     # nothing if unlabelled
        isnothing(ps_labels) ? nothing : ps_labels[pi],
    )
end
```

If your slice type needs to carry additional fields (e.g. metadata specific to your CM), define your own `AbstractCMDataSlice` subtype instead.

### 2. Data accessors (for `GaussianNLL`)

If you use `GaussianNLL`, implement three accessor methods on your slice type:

```julia
SmoreBase._mean(d::MySlice) = d.my_mean_array   # [n_times, n_variables, n_conditions]
SmoreBase._sd(d::MySlice)   = d.my_sd_array     # same shape
SmoreBase._cov(d::MySlice)  = nothing           # or [n_variables, n_variables, n_times, n_conditions]
```

Field names in your slice type are entirely up to you — just return the right array from each accessor.

If you return a `CMDataSlice` from `_sliceCmParamSet`, these accessors are already defined and no further work is needed.

### 3. A `CustomLoss` function (optional)

For custom loss logic, supply a `CustomLoss` whose function receives the slice:

```julia
fn(A_pred::AbstractMatrix, data_slice::AbstractCMDataSlice, condition_idx::Int) -> Float64
```

- `A_pred` — SM prediction at the current optimizer iterate, shape `[n_times × n_variables]`
- `data_slice` — the object returned by your `_sliceCmParamSet`
- `condition_idx` — index of the current condition within `data_slice`

## Worked example

Suppose your CM produces endpoint data (no time axis) with heterogeneous per-condition noise that you want to store separately from the main `CMData`.

```julia
using SmoreBase

# Custom data type: standard CMData + per-condition scale factors
struct ScaledCMData <: AbstractCMData
    base::CMData
    scale::Vector{Float64}   # length n_conditions
end

# Custom slice type that carries the scale for this param-set
struct ScaledCMDataSlice <: AbstractCMDataSlice
    base_slice::CMDataSlice
    scale::Vector{Float64}
end

function SmoreBase._sliceCmParamSet(data::ScaledCMData, pi::Int)
    return ScaledCMDataSlice(
        SmoreBase._sliceCmParamSet(data.base, pi),
        data.scale,
    )
end

SmoreBase.n_cm_param_sets(d::ScaledCMData) = n_cm_param_sets(d.base)
SmoreBase._times(d::ScaledCMData)          = SmoreBase._times(d.base)
SmoreBase._conditions(d::ScaledCMData)     = SmoreBase._conditions(d.base)

# n_variables is not used by the pipeline; only implement it if you want it for inspection code
SmoreBase.n_variables(d::ScaledCMData) = n_variables(d.base)

# Custom loss that applies the per-condition scale
scaled_loss = CustomLoss() do A_pred, slice::ScaledCMDataSlice, ki
    s      = slice.scale[ki]
    μ_view = @view SmoreBase._mean(slice.base_slice)[:, :, ki]  # retrieves stored CM mean; does not compute anything
    σ_view = @view SmoreBase._sd(slice.base_slice)[:, :, ki]    # retrieves stored CM sd; does not compute anything
    return 0.5 * sum((A_pred .- μ_view) .^ 2 ./ (s .* σ_view) .^ 2) +
           0.5 * sum(log.(2π .* (s .* σ_view) .^ 2))
end

# Use exactly like CMData
data = ScaledCMData(
    CMData(μ = rand(10, 2, 3), σ = 0.1 .* ones(10, 2, 3),
           times = collect(0.0:1.0:9.0), variables = 2, conditions = 3),
    [1.0, 2.0, 0.5],
)

problem = SMFitProblem(sm, data, prior; loss = scaled_loss)
result  = fitSurrogate(problem, P0)
```

## Accessor methods

### On your `AbstractCMData` subtype

Only `n_cm_param_sets` must be implemented explicitly. Everything else is either derived or optional:

```julia
SmoreBase.n_cm_param_sets(d::MyData) = size(d.my_array, 5)   # whichever dim holds param-sets
```

`n_conditions` and `n_times` are derived automatically — do not implement them.

If you want `n_variables` for inspection code, add it yourself:

```julia
SmoreBase.n_variables(d::MyData) = size(d.my_array, 2)
```

### On your `AbstractCMDataSlice` subtype

None of the `n_*` accessors are called by the pipeline on slices. Implement them only if
you want them available for your own inspection code:

```julia
SmoreBase.n_times(d::MySlice)      = size(d.my_view, 1)
SmoreBase.n_variables(d::MySlice)  = size(d.my_view, 2)
SmoreBase.n_conditions(d::MySlice) = size(d.my_view, 3)
```

`n_cm_param_sets` is intentionally not defined on `AbstractCMDataSlice` — a slice always represents exactly one param-set.
