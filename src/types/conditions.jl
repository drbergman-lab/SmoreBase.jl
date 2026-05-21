"""
    ConditionSpec(labels)
    ConditionSpec(label::String)
    ConditionSpec()

Categorical labels for the experimental conditions applied to CM runs.
The SM function is responsible for encoding the numeric effect of each condition.

# Examples
```julia
ConditionSpec(["control", "treated"])
ConditionSpec("monotherapy")   # single-condition convenience form
ConditionSpec()                # default single condition "default"
```
"""
struct ConditionSpec
    labels::Vector{String}
end

ConditionSpec(label::String) = ConditionSpec([label])
ConditionSpec() = ConditionSpec(["default"])

Base.length(cs::ConditionSpec) = length(cs.labels)
Base.getindex(cs::ConditionSpec, i) = cs.labels[i]

