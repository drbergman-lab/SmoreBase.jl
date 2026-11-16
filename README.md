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
    mean  = ...,   # [n_cm_param_sets × n_conditions × n_times × n_outputs]
    sd    = ...,   # same shape
    times = t,
)

# Bundle model, data, prior, and loss into a single problem object
prior   = ParameterPrior(lower=[0.0, 0.0], upper=[2.0, 10.0], names=["r", "K"])
problem = SMFitProblem(sm, data, prior)   # loss defaults to GaussianNLL()

# Fit SM parameters (one fit per cm_param_set)
P0  = [0.5 5.0]   # initial guess [n_cm_param_sets × n_sm_params]
fit = fitSurrogate(problem, P0)

# Quantify uncertainty via profile likelihood (explicit index 1: single cm_param_set here;
# omit the index entirely to profile every cm_param_set and get a Vector back)
uq = quantifyUncertainty(ProfileLikelihood(), problem, fit, 1)

# Sample SM predictions within the UQ-defined parameter region
samples = sampleSMPredictions(problem, uq)
```

---

## Roadmap

- [ ] `ODESurrogateModel.y0` — extend to `Dict{String,Vector{Float64}}` for condition-specific initial conditions; relatedly, allow `pre_processor` to alter `y0` itself (not just `p`/`condition`) for conditions that change an initial compartment value (e.g. immunotherapy). Scoped to `ODESurrogateModel` only — `CustomSurrogateModel` closures own their own initial condition — see PRD.md
- [ ] Individual-runs CM data — fit to each replicate rather than mean ± SD (replicate-axis `AbstractCMData` subtype + matching loss); the PCMM extension would then gain a kwarg to choose summary stats vs individual runs — see PRD.md
- [ ] Pipeline persistence — HDF5 serialization/deserialization of `CMData`, `SMFitResult`, `ProfileLikelihoodResult`, `SampledPredictions`
