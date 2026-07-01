# Plotting

SmoreBase ships **Plots.jl recipes** for every result type. Load any Plots backend
to activate `SmoreBasePlotsExt`, then call `plot` on a result:

```julia
using Plots, SmoreBase

plot(SMFitPlot(sm, data, fit_result))   # SM fit over CM data, one panel per output
plot(fit_result)                        # fitted params, colored by convergence
plot(uq_result)                         # profile-likelihood curves
plot(sampled_preds)                     # prediction bands
```

Because these are RecipesBase recipes, the usual Plots attributes pass through for
free — `plot(uq_result; legend = :bottomright, layout = (2, 2))`, etc.
