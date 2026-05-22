using SmoreBase
using Distributions
using RecipesBase
using Test

# ── helpers ────────────────────────────────────────────────────────────────────

# Logistic growth: K / (1 + (K/y0 - 1) * exp(-r*t))
# Parameters: p = [r, K];  y0 (initial condition) = 0.01
_logistic(t, p, _cond) = reshape(
    p[2] ./ (1.0 .+ (p[2] / 0.01 - 1.0) .* exp.(-p[1] .* t)),
    :, 1,
)

# ── ConditionSpec ──────────────────────────────────────────────────────────────

@testset "ConditionSpec" begin
    cs = ConditionSpec(["ctrl", "treated"])
    @test length(cs) == 2
    @test cs[1] == "ctrl"
    @test cs[2] == "treated"

    @test length(ConditionSpec()) == 1
    @test ConditionSpec()[1] == "default"
    @test ConditionSpec("mono")[1] == "mono"
end

# ── ParameterPrior ─────────────────────────────────────────────────────────────

@testset "ParameterPrior" begin
    prior = ParameterPrior([0.0, 0.0], [2.0, 10.0]; names=["r", "K"])
    @test length(prior) == 2
    @test prior.names == ["r", "K"]
    @test SmoreBase._lowerBounds(prior) ≈ [0.0, 0.0]
    @test SmoreBase._upperBounds(prior) ≈ [2.0, 10.0]
    @test prior.distributions[1] isa Uniform
    @test prior.distributions[2] isa Uniform

    # Custom distribution
    prior2 = ParameterPrior(
        [truncated(Normal(1.0, 0.5), 0.0, 2.0), Uniform(1.0, 20.0)],
        ["r", "K"],
    )
    @test SmoreBase._lowerBounds(prior2) ≈ [0.0, 1.0]
    @test SmoreBase._upperBounds(prior2) ≈ [2.0, 20.0]

    # Validation
    @test_throws ArgumentError ParameterPrior([0.0], [2.0, 10.0])
    @test_throws ArgumentError ParameterPrior([2.0, 5.0], [1.0, 10.0])  # lower ≥ upper
end

# ── CMData ─────────────────────────────────────────────────────────────────────

@testset "CMData" begin
    t = collect(0.0:1.0:5.0)   # 6 time points

    # 1-D: times only present → shape (6,)
    data_u = CMData(μ = rand(6), σ = 0.1 .* ones(6), times = t)
    @test data_u isa CMData
    @test n_times(data_u) == 6
    @test n_variables(data_u) == 1
    @test n_conditions(data_u) == 1
    @test n_param_sets(data_u) == 1
    @test data_u.condition_labels == ["c1"]
    @test data_u.variable_labels  == ["y1"]

    # ASCII constructor
    data_a = CMData(mean = rand(6), sd = 0.1 .* ones(6), times = t)
    @test data_a isa CMData

    # Duplicate alias → error
    @test_throws ArgumentError CMData(μ = rand(6), mean = rand(6), σ = ones(6), times = t)
    @test_throws ArgumentError CMData(μ = rand(6), σ = ones(6), sd = ones(6), times = t)

    # Shape mismatch between μ and σ → error
    @test_throws ArgumentError CMData(μ = rand(6), σ = ones(7), times = t)

    # 4-D: all four axis kwargs required for a 4-D array
    data_4d = CMData(
        μ = rand(6, 1, 3, 2), σ = 0.1 .* ones(6, 1, 3, 2), times = t,
        variables  = 1,
        conditions = ["ctrl", "low", "high"],
        param_sets = ["θ₁", "θ₂"],
    )
    @test n_times(data_4d) == 6
    @test n_conditions(data_4d) == 3
    @test n_param_sets(data_4d) == 2

    # 2-D [n_times × n_variables]: variables kwarg present
    data_2d = CMData(μ = rand(6, 2), σ = 0.1 .* ones(6, 2), times = t,
                     variables = ["tumor", "immune"])
    @test n_variables(data_2d) == 2
    @test n_conditions(data_2d) == 1
    @test n_param_sets(data_2d) == 1

    # size mismatch: param_sets says n=1 but array dim has size 3 → error
    @test_throws ArgumentError CMData(
        μ = rand(6, 3), σ = ones(6, 3), times = t,
        param_sets = ["only_one"],
    )
end

@testset "CMData shape promotion" begin
    t = collect(0.0:1.0:5.0)   # n_t = 6

    # 1-D (6,): only times present
    d = CMData(μ = rand(6), σ = ones(6), times = t)
    @test n_times(d) == 6 && n_variables(d) == 1 && n_conditions(d) == 1 && n_param_sets(d) == 1

    # 2-D (6, 2): times + variables — auto-matched by distinct sizes
    d = CMData(μ = rand(6, 2), σ = ones(6, 2), times = t, variables = 2)
    @test n_times(d) == 6 && n_variables(d) == 2 && n_conditions(d) == 1 && n_param_sets(d) == 1

    # 2-D (6, 3): times + conditions
    d = CMData(μ = rand(6, 3), σ = ones(6, 3), times = t,
               conditions = ["ctrl", "low", "high"])
    @test n_times(d) == 6 && n_variables(d) == 1 && n_conditions(d) == 3 && n_param_sets(d) == 1

    # 2-D (6, 4): times + param_sets
    d = CMData(μ = rand(6, 4), σ = ones(6, 4), times = t, param_sets = 4)
    @test n_times(d) == 6 && n_variables(d) == 1 && n_conditions(d) == 1 && n_param_sets(d) == 4

    # 3-D (6, 2, 3): times + variables + conditions — auto-matched
    d = CMData(μ = rand(6, 2, 3), σ = ones(6, 2, 3), times = t,
               variables = ["A", "B"], conditions = ["ctrl", "low", "high"])
    @test n_times(d) == 6 && n_variables(d) == 2 && n_conditions(d) == 3 && n_param_sets(d) == 1

    # 3-D (6, 2, 2): repeated sizes → dim_order required
    d = CMData(μ = rand(6, 2, 2), σ = ones(6, 2, 2), times = t,
               conditions = ["ctrl", "trt"], param_sets = 2,
               dim_order = [:times, :conditions, :param_sets])
    @test n_times(d) == 6 && n_conditions(d) == 2 && n_param_sets(d) == 2

    # Non-canonical input order: [conditions × times] → dim_order resolves + permutes
    d = CMData(μ = rand(3, 6), σ = ones(3, 6), times = t,
               conditions = ["ctrl", "low", "high"],
               dim_order = [:conditions, :times])
    @test n_times(d) == 6 && n_conditions(d) == 3

    # Repeated sizes without dim_order → error
    @test_throws ArgumentError CMData(μ = rand(6, 6), σ = ones(6, 6),
                                      times = collect(1.0:6.0), conditions = 6)

    # Wrong length → error
    @test_throws ArgumentError CMData(μ = rand(5), σ = ones(5), times = t)
    # variables kwarg says 2 but array has 3 in that slot → error
    @test_throws ArgumentError CMData(μ = rand(6, 3), σ = ones(6, 3), times = t, variables = 2)
    # wrong number of non-nothing kwargs for array dimensionality → error
    @test_throws ArgumentError CMData(μ = rand(6, 2), σ = ones(6, 2), times = t)

    # times=nothing: time axis absent, first dim is variables
    d = CMData(μ = rand(3), σ = ones(3), variables = 3)
    @test n_times(d) == 1 && n_variables(d) == 3 && n_conditions(d) == 1 && n_param_sets(d) == 1
    @test d.times === nothing

    # times=nothing: 2-D [variables × conditions]
    d = CMData(μ = rand(2, 4), σ = ones(2, 4), variables = 2, conditions = 4)
    @test n_times(d) == 1 && n_variables(d) == 2 && n_conditions(d) == 4

    # times=nothing: 1-D with param_sets only
    d = CMData(μ = rand(5), σ = ones(5), param_sets = 5)
    @test n_times(d) == 1 && n_variables(d) == 1 && n_conditions(d) == 1 && n_param_sets(d) == 5
end

# ── AnalyticalSurrogateModel / _evaluate ──────────────────────────────────────

@testset "AnalyticalSurrogateModel" begin
    sm = AnalyticalSurrogateModel(fn = _logistic)
    t  = collect(0.0:1.0:5.0)
    p  = [0.5, 5.0]

    A = SmoreBase._evaluate(sm, t, p, "default")
    @test A isa Matrix
    @test size(A) == (6, 1)
    @test all(A .> 0)

    # pre_processor transforms parameters before fn is called
    sm_pre = AnalyticalSurrogateModel(
        fn            = (t, p, c) -> reshape(fill(p[1], length(t)), :, 1),
        pre_processor = (p, c) -> ([p[1] * 2], c),
    )
    A_pre = SmoreBase._evaluate(sm_pre, t, [3.0], "x")
    @test all(A_pre .≈ 6.0)

    # post_processor: multiply by 2
    sm_post = AnalyticalSurrogateModel(
        fn             = (t, p, c) -> ones(length(t), 1),
        post_processor = A -> A .* 2,
    )
    A_post = SmoreBase._evaluate(sm_post, t, p, "x")
    @test all(A_post .≈ 2.0)
end

@testset "ODESurrogateModel without extension" begin
    sm = ODESurrogateModel(
        ode_fn = (du, u, p, t) -> (du[1] = p[1] * u[1] * (1 - u[1] / p[2])),
        y0     = [0.01],
        solver = nothing,
    )
    @test_throws ErrorException SmoreBase._evaluate(sm, [0.0, 1.0], [0.5, 5.0], "default")
end

# ── Loss functions ─────────────────────────────────────────────────────────────

@testset "GaussianNLL" begin
    t    = collect(0.0:1.0:5.0)
    μ_true = _logistic(t, [0.5, 5.0], nothing)  # [6 × 1] matrix (SM output format)
    data = CMData(μ = vec(μ_true), σ = 0.1 .* ones(6), times = t)
    slice = SmoreBase._sliceParamSet(data, 1)

    # Exact prediction → loss = 0.5 * sum(log(2π σ²))
    L_exact = SmoreBase._computeLoss(GaussianNLL(), μ_true, slice, 1)
    @test L_exact isa Float64
    @test L_exact ≈ 0.5 * sum(log.(2π .* (0.1)^2 .* ones(6, 1)))

    # Perturbed prediction → higher loss
    L_perturbed = SmoreBase._computeLoss(GaussianNLL(), μ_true .+ 1.0, slice, 1)
    @test L_perturbed > L_exact
end

@testset "CustomLoss" begin
    t    = collect(0.0:1.0:5.0)
    data = CMData(μ = rand(6), σ = ones(6), times = t)
    slice = SmoreBase._sliceParamSet(data, 1)
    custom = CustomLoss((A, d, ki) -> 42.0)
    @test SmoreBase._computeLoss(custom, zeros(6, 1), slice, 1) == 42.0
end

# ── fitSurrogate ───────────────────────────────────────────────────────────────

@testset "fitSurrogate" begin
    t = collect(0.0:0.5:5.0)   # 11 time points
    p_true = [0.6, 4.0]
    μ_true = _logistic(t, p_true, nothing)
    data   = CMData(μ = vec(μ_true), σ = 0.05 .* ones(length(μ_true)), times = t)

    sm    = AnalyticalSurrogateModel(fn = _logistic)
    prior = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])
    P0    = [0.5 5.0]

    result = fitSurrogate(sm, data, P0, prior)

    @test result isa SMFitResult
    @test size(result.parameters) == (1, 2)
    @test all(result.parameters .>= SmoreBase._lowerBounds(result.prior)')
    @test all(result.parameters .<= SmoreBase._upperBounds(result.prior)')
    @test result.prior.names == ["r", "K"]

    # Fitted parameters should be close to true values
    @test result.parameters[1, 1] ≈ p_true[1] atol = 0.05
    @test result.parameters[1, 2] ≈ p_true[2] atol = 0.1

    # callable executor (map) should give identical result
    result_exec = fitSurrogate(sm, data, P0, prior; executor = map)
    @test result_exec.parameters ≈ result.parameters atol = 1e-4

    # symbol :serial should give identical result
    result_serial = fitSurrogate(sm, data, P0, prior; executor = :serial)
    @test result_serial.parameters ≈ result.parameters atol = 1e-4

    # Dimension validation
    @test_throws ArgumentError fitSurrogate(sm, data, [0.5 5.0 1.0], prior)
end

# ── Profile likelihood ────────────────────────────────────────────────────────

@testset "ProfileLikelihood" begin
    # t range extends to 50 so the curve saturates at K — both r and K are well-identified
    t = collect(0.0:5.0:50.0)
    p_true = [0.6, 4.0]
    μ_true = _logistic(t, p_true, nothing)
    data   = CMData(μ = vec(μ_true), σ = 0.05 .* ones(length(μ_true)), times = t)

    sm     = AnalyticalSurrogateModel(fn = _logistic)
    prior  = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])
    P0     = [0.5 5.0]
    result = fitSurrogate(sm, data, P0, prior)

    method = ProfileLikelihood(n_points = 20, confidence_level = 0.95)
    uq     = SmoreBase._uq(sm, data, result, method)

    @test uq isa ProfileLikelihoodResult
    @test length(uq.profiles) == 2

    for pc in uq.profiles
        # MLE value is always a grid point; its profile LL should match reference_ll to optimizer tolerance
        mle_val    = result.parameters[1, pc.parameter_index]
        mle_idx    = findfirst(≈(mle_val; atol = 1e-10), pc.profile_values)
        @test mle_idx !== nothing
        @test pc.reference_ll - pc.log_likelihoods[mle_idx] < 1e-4

        # Profile LL at any grid point cannot exceed the MLE LL
        @test maximum(pc.log_likelihoods) <= pc.reference_ll + 1e-6

        # Well-identified parameters must have finite CI bounds that bracket the MLE
        fitted_val = result.parameters[1, pc.parameter_index]
        @test pc.ci_lower !== nothing
        @test pc.ci_upper !== nothing
        @test pc.ci_lower < fitted_val < pc.ci_upper
    end

    # Unbounded prior → warning + ArgumentError
    prior_unbounded = ParameterPrior(
        [truncated(Normal(0.6, 1.0), 0.0, Inf), Uniform(0.5, 10.0)],
        ["r", "K"],
    )
    result_ub = fitSurrogate(sm, data, P0, prior_unbounded)
    @test_warn r"unbounded" @test_throws ArgumentError SmoreBase._uq(sm, data, result_ub, ProfileLikelihood(n_points = 5))

    # Single-parameter SM: covers the isempty(free_idx) branch in _profileLL,
    # where fixing the only parameter leaves nothing to re-optimize.
    @testset "single SM parameter" begin
        _exp_decay = (t, p, _) -> reshape(exp.(-p[1] .* t), :, 1)
        t1      = collect(0.0:1.0:10.0)
        p1_true = [0.5]
        μ1      = _exp_decay(t1, p1_true, nothing)
        data1   = CMData(μ = vec(μ1), σ = 0.01 .* ones(length(t1)), times = t1)
        sm1     = AnalyticalSurrogateModel(fn = _exp_decay)
        prior1  = ParameterPrior([0.01], [2.0]; names = ["r"])
        result1 = fitSurrogate(sm1, data1, reshape([0.4], 1, 1), prior1)

        uq1 = SmoreBase._uq(sm1, data1, result1, ProfileLikelihood(n_points = 10))
        @test uq1 isa ProfileLikelihoodResult
        @test length(uq1.profiles) == 1
        pc1 = uq1.profiles[1]
        @test pc1.ci_lower !== nothing
        @test pc1.ci_upper !== nothing
        @test pc1.ci_lower < result1.parameters[1, 1] < pc1.ci_upper
    end
end

# ── sampleSMPredictions ───────────────────────────────────────────────────────

@testset "sampleSMPredictions" begin
    t = collect(0.0:0.5:5.0)
    p_true = [0.6, 4.0]
    μ_true = _logistic(t, p_true, nothing)
    data   = CMData(μ = vec(μ_true), σ = 0.05 .* ones(length(μ_true)), times = t)

    sm     = AnalyticalSurrogateModel(fn = _logistic)
    prior  = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])
    P0     = [0.5 5.0]
    result = fitSurrogate(sm, data, P0, prior)
    uq     = SmoreBase._uq(sm, data, result, ProfileLikelihood(n_points = 20))

    samples = sampleSMPredictions(sm, uq; nSamples = 50)

    @test samples isa SampledPredictions
    @test size(samples.parameters)  == (2, 50)
    @test size(samples.predictions) == (length(t), 1, 50)
    @test samples.times == t

    # Parameters are drawn from the profile grid, so they lie within the prior bounds
    lb = SmoreBase._lowerBounds(result.prior)
    ub = SmoreBase._upperBounds(result.prior)
    for i in 1:2
        @test all(lb[i] .<= samples.parameters[i, :] .<= ub[i])
    end
end

# ── Plotting recipes ───────────────────────────────────────────────────────────

@testset "Plots — SMFitResult" begin
    t      = collect(0.0:0.5:5.0)
    p_true = [0.6, 4.0]
    μ_true = _logistic(t, p_true, nothing)
    data   = CMData(μ = vec(μ_true), σ = 0.05 .* ones(length(μ_true)), times = t)
    sm     = AnalyticalSurrogateModel(fn = _logistic)
    prior  = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])
    result = fitSurrogate(sm, data, [0.5 5.0], prior)

    # SMFitResult: one subplot per parameter × one series per convergence state
    rds = RecipesBase.apply_recipe(Dict{Symbol,Any}(), result)
    @test !isempty(rds)
    # 1 param_set all converged → 2 parameters × 1 converged series = 2 RecipeData
    @test length(rds) == 2

    # SMFitPlot: one series per output variable (fit line + data scatter = 2 series for 1 variable)
    rds2 = RecipesBase.apply_recipe(Dict{Symbol,Any}(), SMFitPlot(sm, data, result))
    @test !isempty(rds2)
    @test length(rds2) == 2   # 1 variable × (fit + data) = 2 series
end

@testset "Plots — ProfileLikelihoodResult / ProfileCurve" begin
    t      = collect(0.0:5.0:50.0)
    p_true = [0.6, 4.0]
    μ_true = _logistic(t, p_true, nothing)
    data   = CMData(μ = vec(μ_true), σ = 0.05 .* ones(length(μ_true)), times = t)
    sm     = AnalyticalSurrogateModel(fn = _logistic)
    prior  = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])
    result = fitSurrogate(sm, data, [0.5 5.0], prior)
    uq     = SmoreBase._uq(sm, data, result, ProfileLikelihood(n_points = 20))

    # ProfileLikelihoodResult: delegates to ProfileCurve recipe → 2 RecipeData (one per parameter)
    rds = RecipesBase.apply_recipe(Dict{Symbol,Any}(), uq)
    @test length(rds) == 2

    # ProfileCurve with both CI bounds: profile + hline + 2 vlines (MLE + lower + upper) = 5 series
    pc  = uq.profiles[1]
    rds_pc = RecipesBase.apply_recipe(Dict{Symbol,Any}(), pc)
    # At minimum: profile LL + threshold hline + MLE vline = 3; CI adds 1–2 more
    @test length(rds_pc) >= 3
    # Both CI bounds present → 5 series total
    if !isnothing(pc.ci_lower) && !isnothing(pc.ci_upper)
        @test length(rds_pc) == 5
    end
end

@testset "Plots — SampledPredictions" begin
    t      = collect(0.0:0.5:5.0)
    p_true = [0.6, 4.0]
    μ_true = _logistic(t, p_true, nothing)
    data   = CMData(μ = vec(μ_true), σ = 0.05 .* ones(length(μ_true)), times = t)
    sm     = AnalyticalSurrogateModel(fn = _logistic)
    prior  = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])
    result = fitSurrogate(sm, data, [0.5 5.0], prior)
    uq     = SmoreBase._uq(sm, data, result, ProfileLikelihood(n_points = 20))
    samples = sampleSMPredictions(sm, uq; nSamples = 30)

    # 1 output variable → 1 ribbon series
    rds = RecipesBase.apply_recipe(Dict{Symbol,Any}(), samples)
    @test length(rds) == 1

    # Missing times → error
    sp_notimes = SampledPredictions(samples.parameters, samples.predictions, nothing)
    @test_throws ErrorException RecipesBase.apply_recipe(Dict{Symbol,Any}(), sp_notimes)
end
