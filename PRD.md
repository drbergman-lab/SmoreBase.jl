# Product Requirements Document — SmoreBase.jl

> **Purpose:** This document defines the complete feature set of SmoreBase in behavioral terms. It is the authoritative answer to "what should this system do?" Read this at the start of any feature session to establish alignment between intent and implementation plan.

---

## Product Overview

**Vision:** SmoreBase provides the foundational layer of the Smore pipeline: train a surrogate model (SM) on complex model (CM) outputs, quantify uncertainty in SM parameters, and propagate that uncertainty to SM predictions.

**Target Users:** Computational modelers who have a slow CM (e.g., an ABM) and want to extract a fast, interpretable surrogate that can be compared to real-world data and analyzed statistically.

**Relationship to MATLAB SMoReParS:** SmoreBase ports and generalizes the SM fitting and profile likelihood components. The "complex model" is not limited to agent-based models — it is any slow simulator.

---

### Feature: CMData

**One-line description:** Structured container for CM simulation summary statistics used to train a surrogate model.

**Priority:** Must-have

**Behavioral specification:**
- `abstract type AbstractCMData end` — base type for CM observation containers
- `abstract type AbstractCMDataSlice <: AbstractCMData end` — base type for a single-param-set view; has no param-set axis (`n_param_sets` not defined). Custom `AbstractCMData` subtypes must implement `_sliceParamSet(data, pi) -> AbstractCMDataSlice`.
- `CMData{T<:Real} <: AbstractCMData` — summary statistics (mean + uncertainty) from CM simulation runs:
  - `μ` — mean observations; canonical shape `[n_times, n_variables, n_conditions, n_param_sets]`; lower-dimensional inputs promoted automatically via keyword constructor
  - `σ` — pointwise standard deviations (same shape as `μ`)
  - `Σ` — optional full covariance `[n_variables, n_variables, n_times, n_conditions, n_param_sets]`; `nothing` means independent observations
  - `times::Union{Nothing,Vector{T}}` — shared time grid
  - `variable_labels::Vector{String}` — names of observable output variables
  - `condition_labels::Vector{String}` — labels for experimental conditions
  - `param_set_labels::Vector{String}` — labels for CM parameter vectors (one SM fit per param_set)
- `CMDataSlice{T<:Real} <: AbstractCMDataSlice` — zero-copy view into a single param-set of a `CMData`; fields `μ`, `σ`, `Σ` are `SubArray` views; created by `_sliceParamSet(data::CMData, pi)`.
- Note: `CMData` holds CM-generated output only. Real-world observational data enters the pipeline in `SmoreFit`, not here.
- The keyword constructor accepts both Unicode and ASCII aliases:
  - `μ` or `mean` — mean observations
  - `σ` or `sd` — standard deviations
  - `Σ` or `cov` — covariance (optional)
  - If both Unicode and ASCII forms are supplied for the same field, throw an `ArgumentError`.
- Constructor validates that `μ` and `σ` have matching shapes; if `Σ` is supplied, validates it is positive semidefinite per time point.

**Supporting types:**
- `ConditionSpec` — experimental conditions as categorical labels:
  - `labels::Vector{String}` — condition labels; the SM function encodes the numeric effect of each condition
  - Convenience constructors: `ConditionSpec("label")`, `ConditionSpec()` (defaults to `["default"]`)
- `ParameterPrior` — SM parameter priors:
  - `distributions::Vector{<:UnivariateDistribution}` — one prior per SM parameter
  - `names::Vector{String}` — parameter names
  - Convenience constructor: `ParameterPrior(lower, upper; names)` wraps pairs into `Uniform` distributions
  - Box bounds derived via `_lowerBounds(prior)` / `_upperBounds(prior)` from distribution support

**Acceptance criteria:**
- `CMData(μ=..., σ=..., times=t)` and `CMData(mean=..., sd=..., times=t)` both construct successfully with default param_set/condition labels.
- Supplying both `μ=` and `mean=` throws a descriptive `ArgumentError`.
- Mismatched `μ` / `σ` shapes throw a descriptive `ArgumentError`.
- `ConditionSpec(["control", "treated"])` stores 2 condition labels.

**Out of scope (v0):**
- Raw cell-level data (variable-length per time point) — defer to a future `CellTableCMData` subtype.

---

### Feature: SurrogateModel Types

**One-line description:** Abstract type hierarchy for surrogate models with ODE and analytical concrete subtypes.

**Priority:** Must-have

**Behavioral specification:**
- `abstract type AbstractSurrogateModel end`
- `ODESurrogateModel{F,Pre,Post,Solve} <: AbstractSurrogateModel`:
  - `ode_fn::F` — in-place ODE RHS: `f!(du, u, p, t)` (SciML convention)
  - `y0::Vector{Float64}` — initial conditions; TODO: extend to `Dict{String,Vector{Float64}}` for condition-specific ICs
  - `solver::Any` — ODE algorithm (e.g., `Tsit5()`); typed `Any` to avoid hard compile-time dependency
  - `output_variables::Union{Nothing,Vector{Int}}` — indices of state variables that correspond to observables; `nothing` means all state variables
  - `pre_processor::Pre` — `Union{Nothing,Function}` applied to inputs before evaluation
  - `post_processor::Post` — `Union{Nothing,Function}` applied to ODE output before returning predictions
  - `custom_solve_fn::Solve` — `Union{Nothing,Function}` — replaces the default ODE solve step if supplied
  - `abstol::Float64 = 1e-6`, `reltol::Float64 = 1e-3`
- `AnalyticalSurrogateModel{F,Pre,Post} <: AbstractSurrogateModel`:
  - `fn::F` — analytical solution: `(t::Vector, p::Vector, condition) -> Matrix{Float64}` where rows are time points, columns are output variables
  - `pre_processor::Pre`, `post_processor::Post`
- Internal dispatch: `_evaluate(sm::AbstractSurrogateModel, t, p, condition) -> Matrix{Float64}`
- ODE extension: `_evaluate` on `ODESurrogateModel` is implemented in `ext/SmoreBaseOrdinaryDiffEqExt.jl`; loading `using OrdinaryDiffEq` activates it. Calling without the extension loaded throws a descriptive error.

**Acceptance criteria:**
- `_evaluate(sm::AnalyticalSurrogateModel, t, p, c)` calls `sm.fn(t, p, c)` and returns a matrix.
- `_evaluate(sm::ODESurrogateModel, t, p, c)` solves the ODE and returns predictions at `t`.
- `pre_processor` is applied before solve (for **both** `ODESurrogateModel` and `AnalyticalSurrogateModel`); `post_processor` is applied after. Signature: `(p, condition) -> (p_new, condition_new)`.
- If `custom_solve_fn` is supplied, it is called instead of the default ODE solver, and receives the **preprocessed** `(p, condition)`.

---

### Feature: Loss Functions

**One-line description:** Pluggable loss functions for comparing SM predictions to CM data.

**Priority:** Must-have

**Behavioral specification:**
- `abstract type AbstractLoss end`
- `GaussianNLL <: AbstractLoss` — default; Gaussian negative log-likelihood:
  - If `Σ` is `nothing`: `NLL = 0.5 * sum((A_pred - μ)² / σ²) + 0.5 * sum(log(2π * σ²))`
  - If `Σ` is supplied: uses the full multivariate Gaussian NLL
- `CustomLoss{F} <: AbstractLoss` — user-supplied loss:
  - `fn::F` — called as `fn(A_pred, data_slice::AbstractCMDataSlice, condition_idx) -> Float64`
- Internal: `_computeLoss(loss, A_pred, data_slice::AbstractCMDataSlice, condition_idx) -> Float64`; dispatches on `AbstractCMDataSlice` — passing unsliced `CMData` is a type error.

**Acceptance criteria:**
- `_computeLoss(GaussianNLL(), A_pred, slice, 1)` returns a scalar, where `slice = _sliceParamSet(data, 1)`.
- `CustomLoss(fn)` where `fn` returns a scalar integrates transparently with `fitSurrogate`.

**Ruled out:**
- A separate `WeightedSSE` type — `GaussianNLL` (without the log-determinant term, or with constant `σ`) already covers this.

---

### Feature: SMFitProblem

**One-line description:** Bundle the surrogate model, data, prior, and loss into a single object passed across the pipeline.

**Priority:** Must-have

**Behavioral specification:**
- `SMFitProblem` — struct with fields `sm::AbstractSurrogateModel`, `data::AbstractCMData`, `prior::ParameterPrior`, `loss::AbstractLoss`
  - Keyword constructor: `SMFitProblem(sm, data, prior; loss = GaussianNLL())`
- `_conditions(data::AbstractCMData) -> ConditionSpec` — derive experimental conditions from data
  - Default implementation: `ConditionSpec()` (single `"default"` condition)
  - `CMData` overrides with its `condition_labels` field
  - Custom `AbstractCMData` subtypes with multiple conditions must override this method
- `SMFitProblem` is passed to `fitSurrogate`, `_uq`, and `sampleSMPredictions` instead of threading `sm`, `data`, `prior`, `loss`, and `conditions` through each call

**Acceptance criteria:**
- `SMFitProblem(sm, data, prior)` stores `GaussianNLL()` as default loss.
- `SMFitProblem(sm, data, prior; loss = CustomLoss(fn))` stores the custom loss.
- `_conditions(data::CMData)` returns a `ConditionSpec` matching `data.condition_labels`.

---

### Feature: Surrogate Model Fitting

**One-line description:** Fit SM parameters to CM summary statistics for each param_set.

**Priority:** Must-have

**Behavioral specification:**
- `fitSurrogate(problem::SMFitProblem, P0; executor, optimOptions) -> SMFitResult`
  - `problem::SMFitProblem` — bundles sm, data, prior, and loss; conditions derived from `problem.data`
  - `P0::AbstractMatrix` — initial parameter guesses `[n_param_sets × n_sm_params]`
  - `executor` — `:serial` (default), `:threads`, `:distributed`, or any callable
  - `optimOptions::NamedTuple = (;)` — forwarded to `Optimization.jl` `solve()`
- Implementation: `Fminbox(LBFGS())` via `OptimizationOptimJL` + `ForwardDiff`; one optimizer call per param_set
- `SMFitResult{T<:Real}`:
  - `parameters::Matrix{T}` — `[n_param_sets × n_sm_params]`
  - `errors::Vector{T}` — objective value per param_set
  - `initial_parameters::Matrix{T}`
  - `prior::ParameterPrior` — carries bounds and parameter names
  - `converged::BitVector`
  - `optim_results::Vector{Any}`

**Acceptance criteria:**
- Returns an `SMFitResult` with `parameters` in `[lb, ub]` for each param_set.
- With `executor=:threads`, results are identical to `:serial` (modulo floating point).
- If a param_set fails to converge, `converged[i] = false` and `parameters[i, :]` contains the best point found.

---

### Feature: UQ of SM Parameters

**One-line description:** Quantify uncertainty in fitted SM parameters via pluggable UQ methods.

**Priority:** Must-have

**Behavioral specification:**
- `abstract type AbstractUQMethod end`
- `ProfileLikelihood <: AbstractUQMethod`:
  - `n_points::Int = 50` — number of grid points per parameter profile
  - `confidence_level::Float64 = 0.95`
  - `bounds::Union{Nothing, ParameterPrior} = nothing` — profile range; defaults to `SMFitResult` bounds
- Internal dispatch: `_uq(problem::SMFitProblem, fitResult, method::AbstractUQMethod; param_set_index) -> ProfileLikelihoodResult`

**Profile likelihood method:**
- For each SM parameter `θ_i`: sweep a grid of `n_points` values anchored at the MLE, fix `θ_i`, re-optimize all other parameters
  - Grid is split proportionally: `n_left` points from `lb_i` to `θ_i*` and `n_right` points from `θ_i*` to `ub_i`; `n_left + n_right - 1 = n_points` (MLE is shared)
  - Each half is scanned outward from the MLE (warm-starting from the previous point)
- Confidence interval by Wilks' theorem: `CI = {θ_i : PL(θ_i) ≥ L* − 0.5 × χ²₁,α}`
  - `L*` = log-likelihood at the MLE (from `fitResult`)
  - For 95% CI: threshold = `L* − 1.92`
- `ProfileCurve{T<:Real}`:
  - `parameter_index::Int`, `parameter_name::String`
  - `profile_values::Vector{T}`, `log_likelihoods::Vector{T}`
  - `ci_lower::Union{Nothing,T}`, `ci_upper::Union{Nothing,T}`
  - `threshold::T`, `reference_ll::T`
- `ProfileLikelihoodResult{T<:Real} <: SMUQResult`:
  - `profiles::Vector{ProfileCurve{T}}`
  - `fit_result::SMFitResult{T}`
  - `cohort_index::Int`
  - `n_profile_points::Int`

**Acceptance criteria:**
- For a well-identified parameter, `ci_lower < fitted_value < ci_upper`.
- For an unidentifiable parameter (flat likelihood), `ci_lower` and/or `ci_upper` is `nothing`.
- The MLE value is always a grid point; its profile LL matches `reference_ll` to optimizer tolerance.

**Future (not in v0):**
- Adaptive profile grid that expands toward the CI boundary.
- `Bootstrap <: AbstractUQMethod`, `MCMC <: AbstractUQMethod`.

---

### Feature: SM Prediction Sampling

**One-line description:** LHS-based Monte Carlo sampling of SM predictions within the UQ-defined parameter region.

**Priority:** Should-have

**Behavioral specification:**
- `sampleSMPredictions(problem::SMFitProblem, uqResult; nSamples, rng) -> SampledPredictions`
  - Samples SM parameters uniformly within the profile-likelihood CI region using LHS (`QuasiMonteCarlo.jl`)
  - Evaluates the SM at each sampled parameter vector; conditions derived from `problem.data`
- `SampledPredictions`:
  - Stores parameter samples, prediction trajectories, and `times` (for standalone plotting)
- This is Monte Carlo propagation of SM parameter uncertainty to output uncertainty — **not** a sensitivity analysis method.

**Acceptance criteria:**
- All sampled parameters lie within the CI bounds from `uqResult`.
- Prediction array has shape `[nSamples × n_times × n_outputs]`.

---

### Feature: Pipeline Persistence (Nextflow-compatible)

**One-line description:** Optional disk serialization of each pipeline step's output.

**Priority:** Should-have

**Motivation:**
Making each step able to write its result to disk and read it back enables Nextflow integration, resumable pipelines, and result sharing.

**Behavioral specification:**
- Each major result type (`CMData`, `SMFitResult`, `ProfileLikelihoodResult`, `SampledPredictions`) must be serializable to and from a standard on-disk format.
- Default format: **HDF5** (`.h5`) via `HDF5.jl` for language-agnostic interop.
- Each pipeline function gains a `save_path::Union{Nothing,AbstractString} = nothing` keyword. When non-`nothing`, the result is written to that path before being returned.
- A symmetric `load_*` function (e.g., `loadSMFitResult(path)`) reads the file and reconstructs the struct.
- Extensibility: `AbstractSerializer` interface for alternate backends (JLD2, Arrow, etc.).

**Acceptance criteria:**
- Round-trip reproduces the result struct exactly (field-by-field equality).
- `save_path = nothing` leaves behavior unchanged.
- HDF5 files are self-describing: dataset names match field names; metadata stored as attributes.

**Out of scope (v0):**
- Streaming / incremental writes during optimization.
- Automatic dependency tracking between files.

---

### Feature: Plotting (RecipesBase extension)

**One-line description:** Plot recipes for every SmoreBase result type via the Plots/RecipesBase backend; Makie users build their own figures from the result types.

**Priority:** Should-have

**Behavioral specification:**

`RecipesBase` is a weak dependency. The Plots extension (`SmoreBasePlotsExt`) activates when `RecipesBase` is loaded.

There is **no Makie extension**. Shipping opinionated `Makie.plot(r) -> Figure` methods conflicted with the composability Makie users expect (they could not customize legend, scale, limits, or layout without reaching into `fig.content`), and the methods duplicated domain knowledge the Plots recipes already encode without adding capability — Makie's rendering, interactivity, and layout power are available to users with or without an extension. Instead, the result types expose their data and `docs/src/plotting.md` shows Makie users how to build each figure directly. See `progress.md` (Drop Makie extension) for the full rationale.

`SMFitPlot` is a plain wrapper struct defined in the main package (not an extension) so the Plots extension can dispatch on it.

| Type | Usage | What it shows |
|------|-------|---------------|
| `SMFitPlot` | `plot(SMFitPlot(sm, data, fit))` | SM fit line overlaid on CM data ± σ; one subplot per output variable |
| `SMFitResult` | `plot(fit_result)` | Fitted parameter values per param_set, colored by convergence; one subplot per SM parameter |
| `ProfileLikelihoodResult` / `ProfileCurve` | `plot(uq_result)` | Profile LL curves with CI threshold, MLE, and CI bound vlines; one subplot per SM parameter |
| `SampledPredictions` | `plot(sampled_preds)` | Quantile ribbon (default 5th–95th) + median line; one subplot per output variable |

**Custom attributes:**
- `band_quantile::Float64 = 0.9` on `SampledPredictions`
- `param_set_index::Int = 1`, `condition_index::Int = 1` on `SMFitPlot`
- Colorblind-friendly colors: `#0072B2` (converged) / `#D55E00` (not converged)

**Testing:**
- `RecipesBase.apply_recipe(Dict{Symbol,Any}(), obj)` exercises each recipe without a backend

**Acceptance criteria:**
- All `apply_recipe` calls return non-empty results without errors.
- Loading SmoreBase without any plotting backend does not error.
- `docs/src/plotting.md` documents the Plots recipes and states there is no Makie extension (result types expose public fields for users who build their own).

---

## Ruled Out / Deferred

- **Raw cell-level CM data in `CMData`**: defer to a future `CellTableCMData` subtype.
- **Separate `WeightedSSE` loss type**: `GaussianNLL` subsumes it.
- **Full `OrdinaryDiffEq` as a direct dependency**: weak dep to keep the package lean.
