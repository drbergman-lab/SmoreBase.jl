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

## Session: SMFitProblem — bundle sm, data, prior, loss (2026-05-22)

### Goal
Thread a single `SMFitProblem` object through `fitSurrogate`, `_uq`, and `sampleSMPredictions`
instead of passing `sm`, `data`, `prior`, `loss`, and `conditions` as separate arguments.
Primary motivation: `_uq` was hardcoding `GaussianNLL()`, silently ignoring any custom loss
used during fitting.

### Key Design Decisions

**`loss` is an input, not a result — same category as `sm` and `data`**
Initial instinct was to store `loss` in `SMFitResult` so `_uq` could read it automatically.
Rejected: `sm` and `data` are already excluded from `SMFitResult` because they're inputs, and
`loss` is in the same category. The consistent solution is to bundle all inputs into one struct.

**`SMFitProblem` rather than adding a `loss` argument to `_uq`**
Bundling all four problem-defining inputs eliminates call-site repetition and makes mismatch
structurally impossible. Adding just `loss` to `_uq` would leave the mismatch risk (user passes
different loss to fit vs. profile).

**`prior` belongs in `SMFitProblem`**
The prior defines the SM parameter space — part of the model definition, not a runtime detail.
`SMFitResult` retains `prior` for result self-containment (useful when loading results from disk).

**`conditions` derived from data via `_conditions(data)`**
`conditions` was always validated against `data.condition_labels` in `fitSurrogate` — i.e., it
was always redundant. `_conditions(d::CMData)` wraps `condition_labels` into a `ConditionSpec`.
Default for custom `AbstractCMData`: `ConditionSpec()` (single "default" condition), matching
the previous default keyword argument. Custom types with multiple conditions must override.

**`_profileLL` keeps explicit `(sm, data_slice, conditions, loss, ...)` args**
`_profileLL` is called in a tight loop from `_uq`. `_uq` extracts `conditions` and `loss` once
before the loop and passes them down. Passing `problem` through the tight loop and re-extracting
fields on every call would be a mild performance smell without benefit.

**`sampleSMPredictions` v0 uses `_conditions(problem.data)[1]` (first condition only)**
Multi-condition sampling remains deferred. Removing the explicit `conditions` kwarg is consistent
— the data is the authoritative source.

### Status
All 25 ProfileLikelihood tests pass (9 new). Full suite green. `feature/sm-fit-problem` ready for review.

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

---

## Session: `pre_processor` fix for ODESurrogateModel (2026-06-15)

### Goal
Two `pre_processor` inconsistencies (from `PREPROCESSOR_HANDOFF.md`):
1. The ODE extension's `_evaluate` never called `_applyPreprocessor`, so a `pre_processor`
   on an `ODESurrogateModel` was silently ignored — violating PRD.md ("applied before solve").
2. Both surrogate-model docstrings documented a 3-arg `(t, p, condition)` signature while
   `_applyPreprocessor` calls `(p, condition)` — a 3-arg implementation would `MethodError`.

### Key Design Decisions

**`custom_solve_fn` receives PRE-processed inputs**
The preprocessor runs at the top of `_evaluate`, before the custom-solve / default-solve
branch, so both paths see the same transformed `(p, condition)`. This matches the analytical
path (where `fn` receives `p_eff`/`c_eff`) and is the natural reading of "applied before solve."

**No runtime guard when both `pre_processor` and `custom_solve_fn` are set**
They compose cleanly (preprocess → custom solve → postprocess) and there are legitimate uses
for both (e.g. log→linear transform + a bespoke solver). A warning would be noise. Instead the
ordering is documented in the `custom_solve_fn` docstring and locked by a test asserting the
custom solver receives the preprocessed `p`.

**Issue 2: docs match code (option a)**
The only `pre_processor` usage in the ecosystem is 2-arg `(p, c)`; nothing passes `t`. So both
docstrings were updated to `(p, condition) -> (p_new, condition_new)` rather than threading `t`
through the implementation. (Sibling `Smore/PRD.md:98` still says `(t, p, c)` — flagged, not
touched, since it's a separate repo.)

### Implementation
- `ext/SmoreBaseOrdinaryDiffEqExt.jl`: apply preprocessor first; refactored to an if-else
  yielding `result`, with a single trailing `return _applyPostprocessor(sm, result)`.
- `src/types/surrogate_model.jl`: corrected both `pre_processor` docstrings; documented that
  `custom_solve_fn` receives preprocessed `(p, condition)`.
- `Project.toml`: added `OrdinaryDiffEq` to the test target (approved) to exercise the ODE path.
- `test/runtests.jl`: new "ODESurrogateModel with extension" testset — basic solve, the Issue 1
  regression (doubled rate → faster growth), and a custom_solve_fn capturing preprocessed `p`.

### Status
Implemented on `feature/ode-preprocessor`. Tests pending run.

---

## Session: Drop Makie extension (2026-06-24)

### Goal
Resolve the Makie plot-composability problem (handoff `HANDOFF-makie-plot-composability.md`)
by **removing the Makie extension entirely** rather than making it composable.

### Problem
`SmoreBaseMakieExt` shipped all-in-one `Makie.plot(r) -> Figure` methods that baked in the
`Figure`, `Axis`, and legend. Users could not customize legend position, axis scale, limits, or
panel layout without reaching into `fig.content`. The "Makie Plot Extensions" session
(2026-05-21) chose ordinary `plot(r) -> Figure` methods over recipes for simplicity; that
simplicity is exactly what cost the composability.

### Decision: remove, don't fix
We weighed three options:
1. **Add a `legend_position` kwarg** — rejected: a kwarg treadmill (then `limits`, then
   `linecolor`, …); wrong altitude.
2. **Convert to a `@recipe` + `plot!` primitive** — the originally-approved design. Rejected
   on implementation: a Makie `@recipe` defined *inside an extension module* generates its
   plotting function (`profilecurve!`) in that module's namespace, unreachable via
   `using SmoreBase`. Exposing it requires predeclaring **and exporting** stub functions in the
   core package, plus a name-collision workaround (`@recipe` lowercases the type name, and
   `ProfileCurve` is already a core data type). That is a lot of machinery in the core package
   for a weak-dep convenience.
3. **Remove the Makie extension; document build-your-own instead** — chosen.

Rationale: the only thing the extension provided was *convenience one-liners that encode where
each result type stores its data*. It did **not** provide nicer rendering, interactivity, or
layout power — those are Makie's and are available to users with or without an extension. The
Plots recipes already encode the same domain knowledge with free attribute/layout passthrough.
Maintaining a second backend — in the form that fights customization — was not worth one
convenience method per type. The durable, zero-maintenance version of that knowledge is
documentation.

### Implementation
- Deleted `ext/SmoreBaseMakieExt.jl` and `HANDOFF-makie-plot-composability.md`.
- `Project.toml`: removed `Makie` from `[weakdeps]`, `[extensions]`, `[compat]`.
- `docs/src/plotting.md` (new): documents the Plots recipes and states there is no Makie
  extension (result types expose public fields for users who roll their own). Registered in
  `docs/make.jl`. NB: an earlier draft inlined full per-type Makie recipes here, but that was
  cut as clutter/maintenance surface for an unsupported path — the build-your-own drawing code
  now lives only where it is actually needed, in the SmoreExamples notebook cells.
- Updated `README.md`, `PRD.md` (Feature: Plotting), `CLAUDE.md` (repo structure, integration
  essentials, weak-dep list), and the `SMFitPlot` docstring to drop Makie-extension claims.

### Decided / do-not-revisit
- **Do not reintroduce a Makie extension.** If a future need arises, prefer extending the
  build-your-own docs over shipping `Figure`-building methods.

### Cross-repo follow-ups
- **SmoreGSA** (`ext/SmoreGSAMakieExt.jl`): same flaw, same fix — extension removed in its own
  repo as part of this work.
- **SmoreExamples**: 4 Pluto notebooks called the removed convenience methods. All 13 call
  sites (`plot(uq)`, `plot(samples)`, `plot(SMFitPlot(...))`, and SmoreGSA's
  `plot(result_efast/morris/custom)`) were rewritten into self-contained `let`-block Makie
  figures and **verified by headless Pluto runs** (`nonidentifiability.jl` → blocks A/B/C incl.
  `SmoreBase._evaluate`; `logistic_growth_pipeline.jl` → block D EFAST/Morris/custom + A/B/C):
  all cells run without error. `single_obs_custom_data.jl` and `replicate_runs_custom_loss.jl`
  use only the profile/band blocks (covered); `cm_posterior_pipeline.jl` hand-builds its figures
  already and was untouched.

### Status
Implemented on `feature/drop-makie-ext`. SmoreBase and SmoreGSA test suites pass; SmoreExamples
notebooks verified headless. Not committed — branches ready for review.

---

## Session: ODE SM refinements + batched profile-likelihood UQ (2026-07-02)

### Goal
Four independent code-review findings, bundled into one branch since they touch adjacent
surrogate-model / UQ code: (b) `custom_solve_fn` doesn't belong on `ODESurrogateModel`, (c) the
default ODE solve anchors `tspan` at `t[1]` instead of `t=0`, (d) `fitSurrogate` requires a
matrix `P0` even for a single guess, (g) `quantifyUncertainty` only profiles one CM param_set at
a time, forcing callers with more than one (SmoreGSA, SmoreFit, SmoreExamples) to hand-roll a
per-param_set fit+profile loop.

### Decisions

**b — new `CustomSolverSurrogateModel` type, not a field.**
`custom_solve_fn` left `ode_fn`, `solver`, `abstol`, `reltol` unused whenever it was set. Split
into its own `AbstractSurrogateModel` subtype (`solve_fn`, `y0`, `pre_processor`,
`post_processor`). Since the custom-solve path never calls into `OrdinaryDiffEq`, its
`_evaluate` lives in the main package next to `AnalyticalSurrogateModel`, not in
`ext/SmoreBaseOrdinaryDiffEqExt.jl` — one fewer thing gated behind the weak dep.

**c — `t0` is a constructor kwarg, not a per-call kwarg.**
`_evaluate`'s signature is fixed across package-extension dispatch, so `t0` had to be a field
(`t0::Float64 = 0.0`) rather than threaded through call sites. `tspan` becomes `(sm.t0,
Float64(t[end]))`; previously `(Float64(t[1]), Float64(t[end]))` silently mis-anchored `y0`
whenever the first observation wasn't at `t=0`.

**e (pulled forward from a related finding) — `names` on `AbstractCMSample`, not `ParameterPrior`.**
While designing (g), we also settled where CM parameter names should live for consumers that
only need labels (not distributions) — SmoreFit's `buildPosterior` was requiring a whole
`ParameterPrior` just to extract `.names`. Added `names::Vector{String}` to `GridCMSample` /
`ScatteredCMSample` (auto-generated `"cm_1", ...` default). SmoreGSA's `runSensitivity` keeps its
separate `cm_prior` argument — there it's load-bearing (inverse-CDF sampling), not just a label
source, so no change needed on that side.

**g — three `quantifyUncertainty` methods, not one method with a `Union` default.**
Rejected an initial design of `cm_param_set_index::Union{Int,Colon} = 1` with a runtime branch
inside one method body — mixes return types (`ProfileLikelihoodResult` vs `Vector`) under one
signature, which is a type-instability smell. Settled on genuine multiple dispatch instead, with
`method` moved to the **first** argument (matching SmoreGSA's `_runSensitivity(method, f, n_cm)`
convention, and marking `method` as the intended extension point for future `AbstractUQMethod`
subtypes):
- `quantifyUncertainty(method, problem, fitResult; executor=:serial)` — no index; **new default**,
  profiles all cm_param_sets, returns `Vector{ProfileLikelihoodResult}`.
- `quantifyUncertainty(method, problem, fitResult, i::Integer)` — single result, opt-in.
- `quantifyUncertainty(method, problem, fitResult, indices::AbstractVector{<:Integer}; executor=:serial)`
  — explicit subset/order; the no-arg form delegates here with `1:n_cm_param_sets(problem.data)`.
- Rejected an explicit `::Colon` dispatch method (`quantifyUncertainty(..., :)`) as redundant —
  the no-arg call already is that path, and unlike `_runSensitivity` (an internal helper with no
  bare-call ergonomic default to preserve), there's no benefit to spelling `:` out.
- Reuses `_resolveExecutor` from `fitting/parallel.jl` so profiling many CM param_sets at once
  gets `:threads`/`:distributed` for free — profiling a CM param_set is `n_params × n_points`
  re-optimizations, more expensive than the fit itself.

### Deferred
Letting `pre_processor` alter `y0` (not just `p`/`condition`) — e.g. a "condition" like
immunotherapy changing an initial compartment value rather than a parameter. Logged as a
"Future (not in v0)" item in PRD.md under SurrogateModel Types; not implemented this session.

### Breaking changes (accepted deliberately — pre-1.0, and touching every call site anyway)
- `ODESurrogateModel` loses `custom_solve_fn`/its `Solve` type param (confirmed unused outside
  SmoreBase by repo-wide grep across SmoreFit/SmoreGSA/SmoreExamples).
- `quantifyUncertainty`'s argument order changes (`method` first) and its default/no-index call
  now returns a `Vector` (length 1 for single-cm_param_set data) instead of a bare
  `ProfileLikelihoodResult`. ~10 call sites in `test/runtests.jl` need updating in this session;
  SmoreExamples' two pipeline notebooks need updating in a follow-up session (they also drop the
  per-param_set `cohort_uq`/`_cohortUQ` closures now that batching is native).
- `buildPosterior` (SmoreFit, separate repo/session) drops its `cm_prior` argument in favor of
  `cm_names` defaulting from `cm_sample.names`.

### Cross-repo follow-ups (tracked, not part of this branch)
- **SmoreGSA**: reorder its own `_runSensitivity(f, n_cm, method)` → `(method, f, n_cm)` — same
  "method first" rationale, independent of everything above.
- **SmoreFit**: `buildPosterior` signature change described above; depends on the `names` field
  landing here first.
- **SmoreExamples**: rewrite `logistic_growth_pipeline.jl` and `cm_posterior_pipeline.jl`
  CM-param_set sections to use batched `fitSurrogate` + `quantifyUncertainty(method, problem,
  result)` in place of the hand-rolled per-param_set loops; fix a wording bug in
  `logistic_growth_pipeline.jl`
  §1 ("treat the logistic equation itself as the SM" → should say **CM**, describing the
  synthetic-data stand-in, not the fitted object).

### Status
In progress on `feature/ode-sm-and-batched-uq`.

---

## Session: Default `P0` from prior medians; `quantifyUncertainty` without a `fitResult` (2026-07-02)

### Goal
Two small ergonomic additions: `fitSurrogate(problem)` should work without an explicit `P0`,
and `quantifyUncertainty` should work without first calling `fitSurrogate` at all.

### Decisions

**Default `P0 = median.(problem.prior.distributions)`**
Median is a reasonable central guess for any `UnivariateDistribution` (not just `Uniform`), and
`Distributions.median` is already available package-wide (`using Distributions` in
`SmoreBase.jl`) with no risk of ambiguity against `Statistics.median` (disjoint method
signatures). Implemented as a thin new method that delegates to the existing
`fitSurrogate(problem, P0::AbstractVector; kwargs...)`, so the broadcast-to-every-cm_param_set
behavior is inherited for free.

**`quantifyUncertainty` no-`fitResult` overloads compute the fit internally, don't discard it**
Added three new methods mirroring the existing `(method, problem, fitResult[, index(es)])`
arities, minus `fitResult`. Each calls `fitSurrogate(problem; executor)` (default-P0 fit) and
delegates to the corresponding `fitResult`-taking method. Dispatch is unambiguous — these differ
from the existing methods purely in arity/argument type (no `SMFitResult` argument at all), not
in a `Union`-typed argument. The internally-computed `fitResult` isn't lost: `ProfileLikelihoodResult`
already carries it in `fit_result`, so callers who only wanted to skip the manual `fitSurrogate`
call can still recover the fit from the result.

**`executor` is reused for both stages, not split into two keywords**
The internal fit and the profiling step share one `executor` keyword. A caller who wants
`:threads` for profiling almost certainly wants it for the (much cheaper) fit too; splitting into
`fit_executor`/`profile_executor` would be a kwarg for a case nobody asked for.

### Status
Implemented on `feature/default-p0-and-uq`. New tests (`fitSurrogate` no-`P0` case,
`quantifyUncertainty without fitResult`) added; full suite green.

---

## Session: `param_set` → `cm_param_set` rename (2026-07-02)

### Goal
Reviewing the batched-UQ work above surfaced a naming ambiguity: SmoreGSA and SmoreFit had been
saying "cohort point" for the same thing SmoreBase calls a `param_set` (one CM parameter
vector). The original 2026-05-19 init session (see above, "cohort" → "param_set" throughout)
already settled on one term over the other, but didn't anticipate this package's fitting/UQ
pipeline also needing to talk about *SM* parameters in the same breath — `fitSurrogate`,
`quantifyUncertainty`, and friends operate over both a CM `param_set` axis and an SM parameter
vector at the same time, and "param_set" alone doesn't say which.

### Decision
Renamed throughout: `param_set(s)` → `cm_param_set(s)`, `ParamSet(s)` → `CmParamSet(s)` (camelCase
internals), including the exported API — `n_param_sets` → `n_cm_param_sets`,
`param_set_index`/`param_set_indices` → `cm_param_set_index`/`cm_param_set_indices`,
`param_set_labels` → `cm_param_set_labels`, `CMData(...; param_sets=...)` →
`CMData(...; cm_param_sets=...)`, `_sliceParamSet` → `_sliceCmParamSet`, etc. Explicit,
deliberate breaking change — not treated as a compat concern (pre-1.0, and every downstream call
site needs touching regardless of how the terminology shakes out). Full test suite green after
the rename.

Superseded, not erased: the original "cohort → param_set" decision above is still correct that
*a single term* was the right call — this just widens that term so it can't be confused with an
SM parameter in the same sentence.

### Cross-repo follow-ups (tracked, not part of this branch)
Same rename needs to land in SmoreFit (`_cohortMLE`/`_meanCohortMLE` internal helpers, plus any
direct calls into SmoreBase's renamed API) and SmoreExamples (`CMData(...; param_sets=...)` call
sites, `n_cohort` locals). SmoreGSA doesn't call the renamed SmoreBase API directly, so no
changes needed there beyond the `_runSensitivity` reorder already tracked above.

### Status
In progress on `feature/ode-sm-and-batched-uq`.
