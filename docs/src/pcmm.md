# PhysiCellModelManager

If your complex model is a PhysiCell agent-based model run through
[PhysiCellModelManager.jl](https://github.com/drbergman-lab/PhysiCellModelManager.jl) (PCMM),
loading PCMM activates `SmoreBasePhysiCellModelManagerExt`, which adds `CMData` and `CMSample`
methods that take PCMM objects directly:

```julia
using SmoreBase, PhysiCellModelManager
initializeModelManager("path/to/project")

sampling = Sampling(1)   # e.g. a GridVariation over two parameters, 4 replicates per point

data = CMData(sampling; cell_types="default", time_unit=:h, obs_times=[24.0, 48.0, 72.0])
# data.μ, data.σ: [3 times × 1 cell type × 1 condition × n_monads]; data.times in hours

cm_sample = CMSample(sampling)
# cm_sample.params: [n_monads × n_varied], one row per monad, one column per varied ABM
# parameter (named in cm_sample.names). Row k is the parameter set of data.μ[:, :, :, k].
```

Each PCMM `Monad` becomes one CM parameter set: `μ` is the replicate mean and `σ` the sample
standard deviation of the population count of each requested cell type.

**Time units.** PhysiCell saves in minutes, so by default `data.times` and `obs_times` are in
minutes. Pass `time_unit` (`:h`, `:d`, … — the symbols PCMM's own plots accept) to get both in
another unit.

**Zero or missing SDs.** Where every replicate agrees (typically t = 0) `σ` is 0, and with a
single replicate it is `NaN`. `CMData` keeps these as they are; `SMFitProblem` rejects them when
the loss is `GaussianNLL`, since the likelihood is undefined there. Leave such times out with
`obs_times`, or fit with a `CustomLoss`.

**Reading speed.** PCMM computes a simulation's population counts by parsing all of its output
snapshots, then saves the counts in that simulation's `summary/` folder. The first `CMData` call on
a set of simulations therefore pays the parsing cost; any later call — say, with different
`obs_times` or `cell_types` — reads the saved counts and is much faster.

```@docs
CMData(::PhysiCellModelManager.AbstractTrial)
CMSample(::PhysiCellModelManager.AbstractTrial)
```
