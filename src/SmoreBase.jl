module SmoreBase

using Distributions
using ForwardDiff
using LinearAlgebra
using Optimization
using OptimizationOptimJL
using QuasiMonteCarlo
using Random
using Statistics

# Types
include("types/cm_data.jl")
include("types/surrogate_model.jl")
include("types/conditions.jl")
include("types/parameter_prior.jl")
include("types/loss.jl")
include("types/results.jl")
include("types/fit_problem.jl")
include("types/cm_sample.jl")
include("types/ci_interp.jl")

# Fitting
include("fitting/objective.jl")
include("fitting/parallel.jl")
include("fitting/fitting.jl")

# Uncertainty quantification
include("profile/ci.jl")
include("profile/profile.jl")

# Prediction sampling
include("sampling.jl")

# SMFitPlot wrapper (plot struct only; recipe in SmoreBasePlotsExt)
include("plots/fit_recipe.jl")

# Exports — types
export AbstractCMData, CMData
export AbstractCMDataSlice, CMDataSlice
export AbstractSurrogateModel, ODESurrogateModel, AnalyticalSurrogateModel
export ConditionSpec, ParameterPrior
export AbstractLoss, GaussianNLL, CustomLoss
export AbstractUQMethod, ProfileLikelihood
export SMFitProblem, SMFitResult, SMUQResult, ProfileLikelihoodResult, ProfileCurve, SampledPredictions

# Exports — CMData accessors
export n_times, n_variables, n_conditions, n_param_sets

# Exports — CM parameter sample layout (shared with SmoreGSA, SmoreFit)
export AbstractCMSample, GridCMSample, ScatteredCMSample, CMSample, reshapeToGrid

# Exports — CI bound interpolation across CM parameter space (shared with SmoreGSA, SmoreFit)
export AbstractCIInterpolator, LinearCIInterp, RBFCIInterp

# Exports — public API
export fitSurrogate, quantifyUncertainty, sampleSMParameters, sampleSMParametersInBounds, sampleSMPredictions

# Exports — plot wrappers
export SMFitPlot

end
