module SmoreBaseMakieExt

using SmoreBase
using Makie
using Statistics

# ── ProfileCurve helper ───────────────────────────────────────────────────────

function _plot_profile_curve!(ax, pc::ProfileCurve)
    mle_val = pc.profile_values[argmax(pc.log_likelihoods)]

    lines!(ax, pc.profile_values, pc.log_likelihoods;
        linewidth = 2, label = "Profile LL")
    hlines!(ax, [pc.threshold];
        linestyle = :dash, color = :gray, linewidth = 1.5, label = "CI threshold")
    vlines!(ax, [mle_val];
        linestyle = :solid, color = :black, linewidth = 1.5, label = "MLE")
    isnothing(pc.ci_lower) || vlines!(ax, [pc.ci_lower];
        linestyle = :dot, color = :red, linewidth = 1.5, label = "CI")
    isnothing(pc.ci_upper) || vlines!(ax, [pc.ci_upper];
        linestyle = :dot, color = :red, linewidth = 1.5, label = "")
end

# ── SMFitPlot ─────────────────────────────────────────────────────────────────

"""
    Makie.plot(fp::SMFitPlot; param_set_index=1, condition_index=1) -> Figure

SM fit overlaid on CM data, one panel per output variable.

Each panel shows a `lines!` series for the SM fit and `errorbars!` + `scatter!`
for the CM data ± pointwise σ.

# Keywords
- `param_set_index::Int = 1` — which param set to display
- `condition_index::Int = 1` — which condition to display

# Example
```julia
using CairoMakie, SmoreBase
fig = Makie.plot(SMFitPlot(sm, data, fit_result))
```
"""
function Makie.plot(fp::SMFitPlot; param_set_index = 1, condition_index = 1, kwargs...)
    ps   = param_set_index
    ci   = condition_index
    data = fp.data
    sm   = fp.sm

    n_v    = n_variables(data)
    times  = data.times
    t_axis = isnothing(times) ? collect(1:size(data.μ, 1)) : times
    x_lbl  = isnothing(times) ? "Index" : "Time"

    cond_label = data.condition_labels[ci]
    p_fitted   = fp.result.parameters[ps, :]
    ŷ          = SmoreBase._evaluate(sm, t_axis, p_fitted, cond_label)

    fig = Figure()
    for v in 1:n_v
        ax = Axis(fig[1, v];
            title  = data.variable_labels[v],
            xlabel = x_lbl,
            ylabel = "Value",
        )
        lines!(ax, t_axis, ŷ[:, v]; linewidth = 2, label = "SM fit")
        errorbars!(ax, t_axis, data.μ[:, v, ci, ps], data.σ[:, v, ci, ps];
            whiskerwidth = 6)
        scatter!(ax, t_axis, data.μ[:, v, ci, ps]; label = "CM data")
        axislegend(ax)
    end
    return fig
end

# ── SMFitResult ───────────────────────────────────────────────────────────────

"""
    Makie.plot(r::SMFitResult) -> Figure

Diagnostic scatter of fitted SM parameter values across all param sets.

One panel per SM parameter. X-axis: param set index; Y-axis: fitted value.
Points are colored by convergence: blue (`#0072B2`) converged, orange (`#D55E00`)
not converged. Legend shown in first panel only.

# Example
```julia
using CairoMakie, SmoreBase
fig = Makie.plot(fit_result)
```
"""
function Makie.plot(r::SMFitResult; kwargs...)
    n_ps, n_p   = size(r.parameters)
    param_names = r.prior.names
    idx_conv    = findall(r.converged)
    idx_notconv = findall(.!r.converged)

    fig = Figure()
    for p_idx in 1:n_p
        ax = Axis(fig[1, p_idx];
            title  = param_names[p_idx],
            xlabel = "Param set",
            ylabel = "Fitted value",
        )
        if !isempty(idx_conv)
            scatter!(ax, idx_conv, r.parameters[idx_conv, p_idx];
                color = "#0072B2",
                label = p_idx == 1 ? "Converged" : "")
        end
        if !isempty(idx_notconv)
            scatter!(ax, idx_notconv, r.parameters[idx_notconv, p_idx];
                color = "#D55E00",
                label = p_idx == 1 ? "Not converged" : "")
        end
        p_idx == 1 && axislegend(ax)
    end
    return fig
end

# ── ProfileCurve ──────────────────────────────────────────────────────────────

"""
    Makie.plot(pc::ProfileCurve) -> Figure

Single profile likelihood panel for one SM parameter.

Shows the profile log-likelihood, Wilks CI threshold (dashed), MLE (solid
vertical), and CI bounds (dotted red verticals, omitted when `nothing`).

# Example
```julia
using CairoMakie, SmoreBase
fig = Makie.plot(uq_result.profiles[1])
```
"""
function Makie.plot(pc::ProfileCurve; kwargs...)
    fig = Figure()
    ax  = Axis(fig[1, 1];
        title  = pc.parameter_name,
        xlabel = pc.parameter_name,
        ylabel = "Log-likelihood",
    )
    _plot_profile_curve!(ax, pc)
    axislegend(ax)
    return fig
end

# ── ProfileLikelihoodResult ───────────────────────────────────────────────────

"""
    Makie.plot(r::ProfileLikelihoodResult) -> Figure

Profile likelihood curves for all SM parameters, one panel per parameter.

Each panel shows the profile LL, Wilks CI threshold, MLE, and CI bounds (when
non-`nothing`). Legend shown in the last panel only to avoid repetition.

# Example
```julia
using CairoMakie, SmoreBase
fig = Makie.plot(uq_result)
```
"""
function Makie.plot(r::ProfileLikelihoodResult; kwargs...)
    n_p = length(r.profiles)
    fig = Figure()
    for (i, pc) in enumerate(r.profiles)
        ax = Axis(fig[1, i];
            title  = pc.parameter_name,
            xlabel = pc.parameter_name,
            ylabel = "Log-likelihood",
        )
        _plot_profile_curve!(ax, pc)
        i == n_p && axislegend(ax)
    end
    return fig
end

# ── SampledPredictions ────────────────────────────────────────────────────────

"""
    Makie.plot(sp::SampledPredictions; band_quantile=0.9) -> Figure

Prediction uncertainty bands from SM parameter sampling, one panel per output.

Each panel shows a quantile ribbon spanning the central `band_quantile` fraction
of samples (default 0.9 → 5th–95th percentile) with the median trajectory on top.

`sp.times` must be non-`nothing`.

# Keywords
- `band_quantile::Float64 = 0.9`

# Example
```julia
using CairoMakie, SmoreBase
fig = Makie.plot(sampled_preds)
fig = Makie.plot(sampled_preds; band_quantile=0.8)
```
"""
function Makie.plot(sp::SampledPredictions; band_quantile = 0.9, kwargs...)
    isnothing(sp.times) && error(
        "SampledPredictions has no stored times; cannot plot without a time axis. " *
        "Ensure `sampleSMPredictions` was called with a `ProfileLikelihoodResult` that has `times` set."
    )

    α_lo  = (1 - band_quantile) / 2
    α_hi  = 1 - α_lo
    times = sp.times
    n_t, n_out, _ = size(sp.predictions)
    band_pct = Int(round(band_quantile * 100))

    fig = Figure()
    for v in 1:n_out
        preds_v = sp.predictions[:, v, :]
        lo  = [quantile(view(preds_v, ti, :), α_lo) for ti in 1:n_t]
        hi  = [quantile(view(preds_v, ti, :), α_hi) for ti in 1:n_t]
        med = [median(view(preds_v,  ti, :))         for ti in 1:n_t]

        ax = Axis(fig[1, v]; xlabel = "Time", ylabel = "Value")
        band!(ax, times, lo, hi; alpha = 0.3, label = "$(band_pct)% band")
        lines!(ax, times, med; linewidth = 2, label = "Median")
        axislegend(ax)
    end
    return fig
end

end # module
