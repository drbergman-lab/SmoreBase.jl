# SmoreBase

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://drbergman-lab.github.io/SmoreBase.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://drbergman-lab.github.io/SmoreBase.jl/dev/)
[![Build Status](https://github.com/drbergman-lab/SmoreBase.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/drbergman-lab/SmoreBase.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/drbergman-lab/SmoreBase.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/drbergman-lab/SmoreBase.jl)

Core library for the [Smore](https://github.com/drbergman-lab/Smore.jl) surrogate modeling ecosystem — a Julia port and generalization of [SMoReParS](https://github.com/drbergman/SMoReParS) (MATLAB). SmoreBase provides the foundational types, fitting, and uncertainty quantification needed to train a surrogate model (SM) on complex model (CM) output and characterize SM parameter uncertainty.

## Quick Start

```julia
using SmoreBase
using OrdinaryDiffEq   # activates ODE-solving extension

# Define a surrogate model (ODE-based example: logistic growth)
sm = ODESurrogateModel(
    ode_fn = (du, u, p, t) -> (du[1] = p[1] * u[1] * (1 - u[1] / p[2])),
    y0 = [0.01],
    solver = Tsit5(),
)

# Supply summary statistics from your complex model runs
data = CMData(
    mean  = ...,   # [n_param_sets × n_conditions × n_times × n_outputs]
    sd    = ...,   # same shape
    times = t,
)

# Fit SM parameters (one fit per param_set)
prior = ParameterPrior(lower=[0.0, 0.0], upper=[2.0, 10.0], names=["r", "K"])
P0    = [0.5 5.0]   # initial guess [n_param_sets × n_sm_params]
fit   = fitSurrogate(sm, data, P0, prior)

# Quantify uncertainty via profile likelihood
uq = _uq(sm, data, fit, ProfileLikelihood())

# Sample SM predictions within the UQ-defined parameter region
samples = sampleSMPredictions(sm, uq)
```

---

## Implementation Status

> For Claude Code sessions: this section is the authoritative record of what has been built. Update it as features are completed. See [PRD.md](PRD.md) for behavioral specifications and [progress.md](progress.md) for decision rationale.

### Completed

- [x] `CMData` / `AbstractCMData` — summary statistics type for CM observations (4-D layout: `[n_param_sets, n_conditions, n_times, n_outputs]`)
- [x] `ConditionSpec`, `ParameterPrior` — supporting types (`ParameterPrior` holds `Distributions.jl` priors; box bounds via `Uniform`)
- [x] `ODESurrogateModel`, `AnalyticalSurrogateModel` — surrogate model types with `_evaluate` dispatch
- [x] ODE extension (`SmoreBaseOrdinaryDiffEqExt`) — ODE solving via `OrdinaryDiffEq.jl`
- [x] `AbstractLoss`, `GaussianNLL`, `CustomLoss` — loss function types
- [x] `fitSurrogate` — fit SM to CM output data via bounded LBFGS optimization (parallel over param_sets)
- [x] `SMFitResult` — result type for SM fitting
- [x] UQ of SM parameters — `ProfileLikelihood` method; `_uq` internal dispatch; MLE-anchored grid with proportional split and outward warm-start
- [x] `ProfileLikelihoodResult`, `ProfileCurve` — result types for UQ
- [x] `sampleSMPredictions` — LHS-based MC sampling within UQ-defined parameter region
- [x] `SampledPredictions` — result type for prediction sampling (stores `times` for standalone plotting)
- [x] Plots extension (`SmoreBasePlotsExt`) — `plot(SMFitPlot(sm, data, fit))`, `plot(fit_result)`, `plot(uq_result)`, `plot(sampled_preds)`; activated by loading `RecipesBase`
- [x] Makie extension (`SmoreBaseMakieExt`) — `Makie.plot(...)` for all result types; activated by loading any Makie backend

### Remaining

- [ ] `ODESurrogateModel.y0` — extend to `Dict{String,Vector{Float64}}` for condition-specific initial conditions
- [ ] Pipeline persistence — HDF5 serialization/deserialization of `CMData`, `SMFitResult`, `ProfileLikelihoodResult`, `SampledPredictions`
