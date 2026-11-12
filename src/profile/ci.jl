# Global switch for degenerate-CI warnings; toggled by `setDegenerateCIWarnings`.
const degenerate_ci_warnings = Ref(true)

"""
    setDegenerateCIWarnings(enabled::Bool) -> Bool

Turn the warnings for degenerate profile-likelihood CIs on or off globally, and return the
previous setting. These warnings are emitted every time `quantifyUncertainty` computes a
CI with a `NaN` bound or zero width (`ci_lower == ci_upper`), and are on by default.

# Arguments
- `enabled` — `true` to emit the warnings, `false` to silence them.

# Returns
The previous setting, so it can be restored.

# Example
```julia
prev = setDegenerateCIWarnings(false)
try
    uq = quantifyUncertainty(problem, ProfileLikelihood())
finally
    setDegenerateCIWarnings(prev)
end
```
"""
function setDegenerateCIWarnings(enabled::Bool)
    prev = degenerate_ci_warnings[]
    degenerate_ci_warnings[] = enabled
    return prev
end

# Compute confidence interval from a profile likelihood curve by linear interpolation.
# Returns (ci_lower, ci_upper); either can be `nothing` if the profile does not cross
# the threshold on that side. Emits a warning (every time, unless silenced with
# `setDegenerateCIWarnings(false)`) when a bound is NaN or both bounds coincide.
function _computeCI(
    profile_values::AbstractVector,
    log_likelihoods::AbstractVector,
    threshold::Real;
    parameter_name::AbstractString = "?",
)
    n = length(profile_values)
    peak_idx = argmax(log_likelihoods)

    ci_lower = _findCrossing(profile_values, log_likelihoods, threshold, 1, peak_idx, :left)
    ci_upper = _findCrossing(profile_values, log_likelihoods, threshold, peak_idx, n, :right)

    if degenerate_ci_warnings[]
        _warnDegenerateCI(ci_lower, ci_upper, parameter_name)
    end
    return ci_lower, ci_upper
end

# Search for the threshold crossing in the range [lo_idx, hi_idx].
# direction :left  → scan from peak leftward (find last crossing below threshold)
# direction :right → scan from peak rightward (find first crossing below threshold)
function _findCrossing(
    vals::AbstractVector,
    lls::AbstractVector,
    threshold::Real,
    lo_idx::Int,
    hi_idx::Int,
    direction::Symbol,
)
    if direction == :left
        # scan from peak toward lower index; find rightmost index where ll < threshold
        for i in (hi_idx - 1):-1:lo_idx
            if lls[i] < threshold && lls[i + 1] >= threshold
                return _linearInterp(vals[i], vals[i + 1], lls[i], lls[i + 1], threshold)
            end
        end
    else  # :right
        # scan from peak toward higher index; find leftmost index where ll < threshold
        for i in lo_idx:(hi_idx - 1)
            if lls[i] >= threshold && lls[i + 1] < threshold
                return _linearInterp(vals[i], vals[i + 1], lls[i], lls[i + 1], threshold)
            end
        end
    end
    return nothing
end

# Linear interpolation to find the x-value where the line (x1,y1)-(x2,y2) equals target.
function _linearInterp(x1, x2, y1, y2, target)
    return x1 + (target - y1) * (x2 - x1) / (y2 - y1)
end

function _warnDegenerateCI(ci_lower, ci_upper, parameter_name)
    hint = "This indicates a degenerate profile, e.g. -Inf log-likelihood at the points " *
           "neighbouring the MLE or a zero-width sweep. Silence with `setDegenerateCIWarnings(false)`."
    if (!isnothing(ci_lower) && isnan(ci_lower)) || (!isnothing(ci_upper) && isnan(ci_upper))
        @warn "Profile-likelihood CI for parameter `$(parameter_name)` has a NaN bound " *
              "(ci_lower = $(ci_lower), ci_upper = $(ci_upper)). " * hint
    elseif !isnothing(ci_lower) && !isnothing(ci_upper) && ci_lower == ci_upper
        @warn "Profile-likelihood CI for parameter `$(parameter_name)` has zero width " *
              "(ci_lower == ci_upper == $(ci_lower)). " * hint
    end
    return nothing
end
