# progress.md — SmoreBase.jl Session Journal

> **Purpose:** Session-level decisions, rejected approaches, and open questions.
> Unlike [PRD.md](PRD.md) (specification) and [README.md](README.md) (completion status), this file captures the *reasoning* behind decisions — things that would otherwise exist only in ended chat history.

---

## Session: Initialization — Package architecture and documentation (2026-05-19)

### Goal
Scaffold the Smore ecosystem: define the monorepo structure (later exploded into separate repos), write documentation, and create stub source files. No implementation code in this session.

### Key Design Decisions Relevant to SmoreBase

**"Complex model" not "ABM"**
The package is named around the concept of a slow, expensive "complex model" (CM) rather than specifically an agent-based model. In all known use cases the CM is an ABM, but the framework is general. Field names, type names, and documentation use "CM" consistently.

**`camelCase` for functions**
Consistent with ModelManager.jl. `camelCase` distinguishes function calls (`fitSurrogate(...)`) from variable names and field accesses, which is especially useful in Julia where both use similar syntax.

**Unicode field names in `CMData`**
`μ`, `σ`, `Σ` for mean, standard deviation, and covariance. Julia supports Unicode identifiers and mathematical notation is standard practice in scientific Julia packages.

**`OrdinaryDiffEq` as a weak dependency (package extension)**
The full `OrdinaryDiffEq.jl` is a large transitive dependency. For users who only use `AnalyticalSurrogateModel`, there is no reason to pay that cost. The ODE-solving logic lives in `ext/SmoreBaseOrdinaryDiffEqExt.jl`.

**`ProfileLikelihood` as one UQ method under `AbstractUQMethod`**
Profile likelihood is the first UQ method implemented (ported from MATLAB SMoReParS), but the API is designed for extensibility. The internal dispatch function `_uq(sm, data, fitResult, method; ...)` takes the method as a type argument.

**Confidence interval formula: Wilks' theorem**
`CI = {θ_i : PL(θ_i) ≥ L* − 0.5 × χ²₁,α}` — verified against Wilks (1938). For 95%: threshold = `L* − 1.92`.

**`GaussianNLL` only (no separate `WeightedSSE`)**
`WeightedSSE` is a special case of `GaussianNLL` (dropping the log-determinant term when `σ` does not depend on `p`). A separate type would create ambiguity. Power users can supply a `CustomLoss`.

**`sampleSMPredictions` is not a sensitivity method**
LHS-based Monte Carlo sampling within the profile-likelihood CI region is uncertainty propagation, not global sensitivity analysis. It lives in SmoreBase as a utility, not in SmoreGSA.

### Open Questions (at time of init)

- **Higher-level pipeline API**: users should call something like `runSmoreBase(sm, data, ...)` that orchestrates fitting + UQ. Deferred to implementation session.
- **`CMData` shape for multiple cohorts and conditions**: resolved in implementation (4-D `[n_param_sets, n_conditions, n_times, n_outputs]`).

---

## Session: SmoreBase Implementation (2026-05-19)

### Goal
Implement all SmoreBase stub files (types, fitting, profile likelihood, sampling, ODE extension) and tests.

### Key Design Decisions

**Conditions are categorical labels only**
`ConditionSpec` wraps `Vector{String}`. The SM function encodes the numeric effect of each condition internally. Eliminates the `ConditionSpec.values::Matrix` field from the original plan.

**`ParameterBounds` → `ParameterPrior`**
Generalized from box bounds to a `Vector{<:UnivariateDistribution}`. Box bounds are represented as `Uniform(lb, ub)`. Convenience constructor `ParameterPrior(lower, upper; names)` wraps pairs into `Uniform`. Optimization bounds derived from `support(d)` via `_lowerBounds`/`_upperBounds`. This makes the type directly useful for SmoreFit posterior inference.

**"cohort" → "param_set" throughout**
More precise terminology: one param_set = one CM parameter vector whose runs generated training data for the SM. `CMData` axis 1 is `n_param_sets`; field name `param_set_labels`.

**CMData canonical shape: 4-D `[n_param_sets, n_conditions, n_times, n_outputs]`**
2-D and 3-D inputs are promoted automatically in the keyword constructor.

**`ForwardDiff` added as a direct dependency**
`Optimization.jl` v5 requires an explicit ADType (`AutoForwardDiff()`) for gradient-based optimization — there is no implicit fallback. `ForwardDiff` was already a transitive dependency; adding it to `Project.toml` does not install any new packages.

**Profile likelihood: fixed-parameter optimization via projection**
`_profileLL` builds a reduced `(n_params-1)`-vector objective: captures the full objective in a closure, projects free parameters back into the full space, fixes `p[fixed_idx] = fixed_val`. The inner closure correctly handles `ForwardDiff.Dual` elements by using `T = eltype(p_free)`.

**`sampleSMPredictions` v0: first condition only**
LHS sampling evaluates the SM at `conditions[1]` only. Multi-condition sampling deferred.

**QuasiMonteCarlo 0.3.x: `LatinHypercubeSample(rng=rng)` support**
The `rng` keyword is supported in the installed QMC version via `@kwdef`. For reproducibility, callers should set the `rng` keyword.

### Status
All 11 stub files implemented, `sampling.jl` added, `SmoreBase.jl` updated. All tests pass.

---

## Session: MLE-anchored profile grid (2026-05-20)

### Problem
The profile likelihood grid was built as a regular `range(lb, ub; length=n_points)`, which almost never includes the exact MLE value. When a parameter is sharply identified, the LL can drop below the CI threshold within a single grid step of the MLE. If no grid point has `ll >= threshold`, `_computeCI` returns `nothing` for both bounds — a false unidentifiability result.

Reported symptom: `ci_lower = nothing, ci_upper = nothing` when profiling K in a logistic growth model with T_final=50 (dynamics reach carrying capacity, so K is well-identified).

### Design Decisions

**MLE-anchored grid with proportional split**
Always include `p_mle[i]` as a grid point. Split the remaining `n_points - 1` points in proportion to the distance from the MLE to each boundary:
- `frac_left = (mle_val - lb) / (ub - lb)`
- `n_left = max(1, round(Int, frac_left * (n_points - 1)) + 1)` (includes MLE)
- `n_right = n_points - n_left + 1` (includes MLE; deduped when concatenating)

Equal split (`ceil(n_points/2)`) was rejected: when the MLE is near a boundary, it wastes most points on the infeasible side.

**Outward warm-start**
Each half is scanned outward from the MLE (left half: MLE → lb; right half: MLE → ub), warm-starting the inner optimizer from the previous grid point.

**Test data range extended to t=50**
The ProfileLikelihood test previously used t ∈ [0, 5], which leaves K completely unidentifiable. Changed to t ∈ [0:5:50] so K is visible in the data.

### Status
All SmoreBase tests pass (16 assertions in ProfileLikelihood).

---

## Session: Plotting Recipes (RecipesBase.jl) (2026-05-21)

### Goal
Add backend-agnostic plot recipes to SmoreBase so users can assess every pipeline stage with `plot(result)`.

### Key Design Decisions

**RecipesBase as a weak dep (package extension)**
`RecipesBase` is a weak dep activated by loading any Plots.jl-compatible backend. The extension `SmoreBasePlotsExt` registers the recipes. This keeps SmoreBase lean for users who only use Makie or no plotting.

**`SMFitPlot` wrapper struct stays in main package**
`SMFitPlot` is a plain struct that both the Plots and Makie extensions dispatch on. If it lived in an extension, users of the other backend would need to load a backend they don't use just to construct it. It lives in `src/plots/fit_recipe.jl` and is exported from SmoreBase.

**Standalone `plot(SMFitResult)` for parameter diagnostics**
Shows fitted values on y, param_set index on x, one subplot per SM parameter, colored by convergence. Two separate series handle the two states; empty series are omitted. Colorblind-friendly colors: `#0072B2` (converged) / `#D55E00` (not converged) from the Wong/Paul Tol palette.

**`SampledPredictions.times` field added**
`sampleSMPredictions` already uses `uqResult.times` internally; adding it to the struct is a minimal, zero-breaking change that enables `plot(sampled)` to work standalone.

**`ProfileLikelihoodResult` delegates to `ProfileCurve` recipe**
The top-level recipe sets `layout := (1, n_params)` and delegates each panel to the `ProfileCurve` recipe via `@series`.

### Status
All 9 new test sets pass, no regressions.

---

## Session: Makie Plot Extensions (2026-05-21)

### Goal
Add optional Makie ecosystem plot support as a Julia package extension, mirroring every RecipesBase recipe.

### Key Design Decisions

**`Makie` as weak dep, not `MakieCore`**
`MakieCore` only provides the `@recipe` macro and abstract plot types. The extensions create multi-panel figures using `Figure`, `Axis`, `lines!`, `scatter!`, `band!`, etc. — all in `Makie`. All Makie backends (CairoMakie, GLMakie, WGLMakie) depend on and load `Makie`, so the extension fires for any backend.

**Single extension (`SmoreBaseMakieExt`), not two**
An earlier idea proposed separate `MakieCoreExt` and `MakieExt`. Rejected: both would fire simultaneously, creating redundancy.

**`Makie.plot(result)` methods, not `@recipe` types**
Ordinary Julia methods `Makie.plot(r::SomeType; kwargs...) -> Figure` rather than `@recipe`-based custom plot types. Simpler and directly mirrors the RecipesBase recipes.

**`_evaluate` accessed via `SmoreBase._evaluate` in the extension**
`using SmoreBase` in the extension gives only exported symbols. The fit recipe calls `SmoreBase._evaluate(...)` with explicit module prefix, consistent with the ODE extension pattern.

### Status
All files written. No tests added (Makie is a large dependency, not suitable for the CI test suite). Ready for review.

---

## Session: CMDataSlice — data abstraction refactor (2026-05-22)

### Goal
Remove `param_set_idx` from the inner optimization loop. `_buildObjective` and `_computeLoss` had no business knowing where in the param-set dimension their data lived — they should receive only what they need.

### Key Design Decisions

**`AbstractCMDataSlice <: AbstractCMData` type hierarchy**
A slice is still a kind of CM data container (same logical interface, reduced by one axis), so subtyping `AbstractCMData` makes sense. It also lets `_computeLoss` dispatch specifically on `AbstractCMDataSlice`, making it a compile-time error to accidentally pass unsliced data. `n_param_sets` is intentionally not defined on the slice type.

**`CMDataSlice` uses `SubArray` views, not copies**
`@view(data.μ[:, :, :, pi])` is a pointer + metadata — 8 bytes, not a data copy. Capturing a `CMDataSlice` in an optimization closure is zero-cost regardless of how large the parent `CMData` is.

**Slice at the earliest caller that's already scoped to one param-set**
- `_fitOneParamSet` already receives `param_set_idx` → slices there, passes `data_slice` to `_buildObjective`
- `_uq` already receives `param_set_index` → slices once before the inner grid loop, passes `data_slice` to `_profileLL`
- `_buildObjective` and `_profileLL` now take `AbstractCMDataSlice` directly; `param_set_idx` is gone from their signatures

**`_uq` still carries `param_set_index` for `fitResult` access**
`fitResult.parameters[param_set_index, :]` and `fitResult.errors[param_set_index]` are struct-of-arrays accesses that would require a `FitResultSlice` type to eliminate. That refactor is deferred; the `param_set_index` in `_uq` is setup code (runs once, not in the hot loop) and the discordance with the now-sliced `data` is acceptable.

**Array-of-structs not pursued**
Fully eliminating all param-set indexing would require restructuring `SMFitResult` and `ProfileLikelihoodResult` into arrays of per-param-set structs. In Julia, struct-of-arrays is the correct layout for numerical computation (cache efficiency, SIMD). The current matrix layout for result types is the right choice.

**`CustomLoss.fn` signature updated**
`fn(A_pred, data_slice::AbstractCMDataSlice, condition_idx)` — the `param_set_idx` argument is gone. Custom loss authors receive a pre-sliced view and index only by condition.

**`docs/src/custom_data.md` added**
How-to guide for users implementing custom `AbstractCMData` subtypes: explains `_sliceParamSet`, the extension point for `CustomLoss`, and includes a worked example with a custom slice type.

### Status
All 100 tests pass. `feature/cm-data-slice` branch ready for review.

---

## Session: RecipesBase → Package Extension (2026-05-21)

### Goal
Move the RecipesBase plot recipes from a direct dependency to a package extension (`SmoreBasePlotsExt`), so Plots.jl and Makie are uniformly optional.

### Key Design Decisions

**`SMFitPlot` stays in the main package**
The struct is constructed before any plot function is called — both the Plots and Makie extensions dispatch on it. Moving it to an extension would require users to load RecipesBase just to construct it.

**`using RecipesBase` removed from `SmoreBase.jl`**
Recipe registration happens inside the extension when the user loads RecipesBase. Tests already `using RecipesBase` at the top, so the extension fires automatically during test runs.

### Status
All files written. Existing tests unchanged — `using RecipesBase` in each test file triggers the extension.
