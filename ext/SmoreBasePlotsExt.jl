module SmoreBasePlotsExt

using SmoreBase
using RecipesBase
using Statistics

# ── SMFitPlot recipe ──────────────────────────────────────────────────────────

@recipe function f(fp::SMFitPlot)
    sm     = fp.sm
    data   = fp.data
    result = fp.result

    ps = pop!(plotattributes, :param_set_index, 1)
    ci = pop!(plotattributes, :condition_index,  1)

    n_v    = n_variables(data)
    times  = data.times
    t_axis = isnothing(times) ? collect(1:size(data.μ, 1)) : times
    x_lbl  = isnothing(times) ? "Index" : "Time"

    layout := (1, n_v)

    cond_label = data.condition_labels[ci]
    p_fitted   = result.parameters[ps, :]
    ŷ          = SmoreBase._evaluate(sm, t_axis, p_fitted, cond_label)

    for v in 1:n_v
        @series begin
            subplot    := v
            seriestype := :path
            label      := "SM fit"
            linewidth  := 2
            title      := data.variable_labels[v]
            xlabel     := x_lbl
            ylabel     := "Value"
            t_axis, ŷ[:, v]
        end

        @series begin
            subplot    := v
            seriestype := :scatter
            label      := "CM data"
            yerror     := data.σ[:, v, ci, ps]
            title      := data.variable_labels[v]
            xlabel     := x_lbl
            ylabel     := "Value"
            t_axis, data.μ[:, v, ci, ps]
        end
    end
end

# ── SMFitResult diagnostic recipe ─────────────────────────────────────────────

"""
    plot(result::SMFitResult)

Diagnostic scatter of fitted SM parameter values across all param sets.

One subplot per SM parameter. X-axis: param set index (1…n_param_sets);
Y-axis: fitted parameter value. Points are colored by convergence status:
- Blue (`#0072B2`) — converged
- Orange (`#D55E00`) — not converged

Legend labels are shown in the first subplot only to avoid repetition.

# Example
```julia
using Plots
plot(fit_result)
```
"""
@recipe function f(r::SMFitResult)
    n_ps, n_p   = size(r.parameters)
    param_names = r.prior.names

    layout := (1, n_p)

    idx_conv    = findall(r.converged)
    idx_notconv = findall(.!r.converged)

    for p_idx in 1:n_p
        if !isempty(idx_conv)
            @series begin
                subplot           := p_idx
                seriestype        := :scatter
                title             := param_names[p_idx]
                xlabel            := "Param set"
                ylabel            := "Fitted value"
                label             := p_idx == 1 ? "Converged" : ""
                markercolor       := "#0072B2"
                markerstrokecolor := "#0072B2"
                idx_conv, r.parameters[idx_conv, p_idx]
            end
        end

        if !isempty(idx_notconv)
            @series begin
                subplot           := p_idx
                seriestype        := :scatter
                title             := param_names[p_idx]
                xlabel            := "Param set"
                ylabel            := "Fitted value"
                label             := p_idx == 1 ? "Not converged" : ""
                markercolor       := "#D55E00"
                markerstrokecolor := "#D55E00"
                idx_notconv, r.parameters[idx_notconv, p_idx]
            end
        end
    end
end

# ── ProfileLikelihoodResult recipe ────────────────────────────────────────────

"""
    plot(result::ProfileLikelihoodResult)

Profile likelihood curves for all SM parameters.

Produces one subplot per profiled parameter by delegating to the `ProfileCurve`
recipe. Each panel shows the profile log-likelihood, the Wilks CI threshold
(dashed), the MLE (solid vertical line), and the CI bounds (dotted vertical
lines, omitted when `nothing`).

# Example
```julia
using Plots
plot(uq_result)
```
"""
@recipe function f(r::ProfileLikelihoodResult)
    n_p = length(r.profiles)
    layout := (1, n_p)

    for (i, pc) in enumerate(r.profiles)
        @series begin
            subplot := i
            pc
        end
    end
end

# ── ProfileCurve recipe ────────────────────────────────────────────────────────

"""
    plot(curve::ProfileCurve)

Single profile likelihood panel for one SM parameter.

Shows the profile log-likelihood curve, a dashed horizontal line at the Wilks
CI threshold, a solid vertical line at the MLE, and dotted vertical lines at the
CI bounds (omitted if `ci_lower` or `ci_upper` is `nothing`).

# Example
```julia
using Plots
plot(uq_result.profiles[1])
```
"""
@recipe function f(pc::ProfileCurve)
    mle_idx = argmax(pc.log_likelihoods)
    mle_val = pc.profile_values[mle_idx]

    title  := pc.parameter_name
    xlabel := pc.parameter_name
    ylabel := "Log-likelihood"

    @series begin
        seriestype := :path
        linewidth  := 2
        label      := "Profile LL"
        pc.profile_values, pc.log_likelihoods
    end

    @series begin
        seriestype := :hline
        linestyle  := :dash
        linecolor  := :gray
        linewidth  := 1.5
        label      := "CI threshold"
        [pc.threshold]
    end

    @series begin
        seriestype := :vline
        linestyle  := :solid
        linecolor  := :black
        linewidth  := 1.5
        label      := "MLE"
        [mle_val]
    end

    if !isnothing(pc.ci_lower)
        @series begin
            seriestype := :vline
            linestyle  := :dot
            linecolor  := :red
            linewidth  := 1.5
            label      := "CI"
            [pc.ci_lower]
        end
    end

    if !isnothing(pc.ci_upper)
        @series begin
            seriestype := :vline
            linestyle  := :dot
            linecolor  := :red
            linewidth  := 1.5
            label      := ""
            [pc.ci_upper]
        end
    end
end

# ── SampledPredictions recipe ─────────────────────────────────────────────────

"""
    plot(sp::SampledPredictions)

Prediction uncertainty bands from SM parameter sampling.

One subplot per output variable. Each subplot shows a quantile ribbon spanning
the `band_quantile` central fraction of samples (default: 0.9 → 5th–95th
percentile), with the median trajectory plotted on top.

`sp.times` must be non-`nothing`; `sampleSMPredictions` populates this field
automatically.

# Plot attributes
- `band_quantile::Float64 = 0.9`

# Example
```julia
using Plots
samples = sampleSMPredictions(sm, uq_result; nSamples=200)
plot(samples)
plot(samples; band_quantile=0.8)
```
"""
@recipe function f(sp::SampledPredictions)
    isnothing(sp.times) && error(
        "SampledPredictions has no stored times; cannot plot without a time axis. " *
        "Ensure `sampleSMPredictions` was called with a `ProfileLikelihoodResult` that has `times` set."
    )

    bq   = pop!(plotattributes, :band_quantile, 0.9)
    α_lo = (1 - bq) / 2
    α_hi = 1 - α_lo

    times            = sp.times
    n_t, n_out, n_s  = size(sp.predictions)

    layout := (1, n_out)

    for v in 1:n_out
        preds_v = sp.predictions[:, v, :]

        lo  = [quantile(view(preds_v, ti, :), α_lo) for ti in 1:n_t]
        hi  = [quantile(view(preds_v, ti, :), α_hi) for ti in 1:n_t]
        med = [median(view(preds_v, ti, :))          for ti in 1:n_t]

        band_pct = Int(round(bq * 100))

        @series begin
            subplot    := v
            seriestype := :path
            ribbon     := (med .- lo, hi .- med)
            fillalpha  := 0.3
            linewidth  := 2
            label      := "$(band_pct)% band"
            xlabel     := "Time"
            ylabel     := "Value"
            times, med
        end
    end
end

end # module
