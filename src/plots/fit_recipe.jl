"""
    SMFitPlot

Plot wrapper for visualizing a surrogate model fit overlaid on CM data.

Construct with `SMFitPlot(sm, data, fit_result)` and pass to `plot()`.

One subplot is produced per output variable. Each subplot overlays the SM fit
line on the CM data scatter ± pointwise σ error bars.

# Plot attributes
- `param_set_index::Int = 1` — which param set to display
- `condition_index::Int = 1` — which condition to display

# Example
```julia
using Plots
plot(SMFitPlot(sm, data, fit_result))
plot(SMFitPlot(sm, data, fit_result); param_set_index=2)
```
"""
struct SMFitPlot{SM<:AbstractSurrogateModel, D<:AbstractCMData, R<:SMFitResult}
    sm     :: SM
    data   :: D
    result :: R
end
