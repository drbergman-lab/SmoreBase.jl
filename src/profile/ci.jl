# Compute confidence interval from a profile likelihood curve by linear interpolation.
# Returns (ci_lower, ci_upper); either can be `nothing` if the profile does not cross
# the threshold on that side.
function _computeCI(
    profile_values::AbstractVector,
    log_likelihoods::AbstractVector,
    threshold::Real,
)
    n = length(profile_values)
    peak_idx = argmax(log_likelihoods)

    ci_lower = _findCrossing(profile_values, log_likelihoods, threshold, 1, peak_idx, :left)
    ci_upper = _findCrossing(profile_values, log_likelihoods, threshold, peak_idx, n, :right)

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
