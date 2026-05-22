# Custom CM Data Types

SmoreBase ships with `CMData`, a general-purpose container for complex-model (CM) summary statistics. For cases where `CMData` does not fit — e.g. your CM produces structured outputs that don't map cleanly onto a `[n_times × n_variables × n_conditions × n_param_sets]` array — you can subtype `AbstractCMData` and teach SmoreBase how to use your type.

## Required interface

Always implement:

- `_sliceParamSet` — required for any custom `AbstractCMData` subtype
- `n_times`, `n_variables`, `n_conditions`, `n_param_sets` — required on your `AbstractCMData`
  subtype; `fitSurrogate` calls these for input validation before fitting begins
- `_mean`, `_sd`, `_cov` — required on your `AbstractCMDataSlice` subtype if using `GaussianNLL`

Optionally implement:

- `CustomLoss` — for fully custom loss logic that doesn't use `GaussianNLL`

### 1. `_sliceParamSet`

```julia
SmoreBase._sliceParamSet(data::MyData, pi::Int) -> AbstractCMDataSlice
```

Return an `AbstractCMDataSlice` containing only the data for param-set `pi`. Some guidelines:

- **Array slices** (sub-arrays indexed by `pi`): use `@view` to avoid copying — each param-set gets its own slice, so copying would duplicate your entire dataset once per param-set.
- **Whole arrays** (like `times` or label vectors): pass directly. Julia passes arrays by reference, so no copy occurs.

The simplest option is to return a `CMDataSlice` built from views into your own arrays:

```julia
function SmoreBase._sliceParamSet(data::MyData, pi::Int)
    ps_labels = SmoreBase._paramSetLabels(data)
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

If you return a `CMDataSlice` from `_sliceParamSet`, these accessors are already defined and no further work is needed.

### 3. A `CustomLoss` function (optional)

For custom loss logic, supply a `CustomLoss` whose function receives the slice:

```julia
fn(A_pred::AbstractMatrix, data_slice::AbstractCMDataSlice, condition_idx::Int) -> Float64
```

- `A_pred` — SM prediction at the current optimizer iterate, shape `[n_times × n_variables]`
- `data_slice` — the object returned by your `_sliceParamSet`
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

function SmoreBase._sliceParamSet(data::ScaledCMData, pi::Int)
    return ScaledCMDataSlice(
        SmoreBase._sliceParamSet(data.base, pi),
        data.scale,
    )
end

SmoreBase.n_times(d::ScaledCMData)      = n_times(d.base)
SmoreBase.n_variables(d::ScaledCMData)  = n_variables(d.base)
SmoreBase.n_conditions(d::ScaledCMData) = n_conditions(d.base)
SmoreBase.n_param_sets(d::ScaledCMData) = n_param_sets(d.base)

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

result = fitSurrogate(sm, data, P0, prior; loss = scaled_loss)
```

## Accessor methods

### On your `AbstractCMData` subtype (required)

`fitSurrogate` calls four shape accessors on the data object before fitting. Implement all four,
pointing each at the appropriate dimension of your underlying storage:

```julia
SmoreBase.n_times(d::MyData)      = size(d.my_array, 1)
SmoreBase.n_variables(d::MyData)  = size(d.my_array, 2)
SmoreBase.n_conditions(d::MyData) = size(d.my_array, 3)
SmoreBase.n_param_sets(d::MyData) = size(d.my_array, 4)   # or whichever dim holds param-sets
```

If your type uses a non-standard layout (e.g. param-sets in dim 5), adjust accordingly.

### On your `AbstractCMDataSlice` subtype (optional)

The slice accessors are not called by the fitting or UQ pipeline, but are useful if you
want `n_times`, `n_variables`, `n_conditions` to work on slices for your own inspection code:

```julia
SmoreBase.n_times(d::MySlice)      = size(d.my_view, 1)
SmoreBase.n_variables(d::MySlice)  = size(d.my_view, 2)
SmoreBase.n_conditions(d::MySlice) = size(d.my_view, 3)
```

`n_param_sets` is intentionally not defined on `AbstractCMDataSlice` — a slice always represents exactly one param-set.
