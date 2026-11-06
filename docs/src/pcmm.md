# CMData from PhysiCellModelManager

If your complex model is a PhysiCell agent-based model run through
[PhysiCellModelManager.jl](https://github.com/drbergman-lab/PhysiCellModelManager.jl) (PCMM),
[`cmDataFromPCMM`](@ref) builds the `CMData` directly from a PCMM trial. Loading PCMM activates
the `SmoreBasePhysiCellModelManagerExt` extension:

```julia
using SmoreBase, PhysiCellModelManager
initializeModelManager("path/to/project")

sampling = Sampling(1)   # e.g. a GridVariation over two parameters, 4 replicates per point
out = cmDataFromPCMM(sampling; cell_types="default", obs_times=[24.0, 48.0, 72.0])

out.data        # CMData [3 times × 1 cell type × 1 condition × n_monads]
out.cm_sample   # GridCMSample of the varied parameters, rows aligned with out.data
out.monad_ids   # PCMM monad behind each CM parameter set
```

Each PCMM `Monad` becomes one CM parameter set: `μ` is the replicate mean and `σ` the sample
standard deviation of the population count of each requested cell type. Times are converted
from PhysiCell minutes to hours by default (`time_scale=1/60`).

A standard deviation of 0 (typically at t = 0, where every replicate starts from the same initial
condition) is an error rather than an infinite likelihood: leave that time out with `obs_times`,
or pass `sd_floor`.

The first call on a simulation parses all of its output snapshots (about 1 s each); PCMM caches
the counts, so later calls are fast.
