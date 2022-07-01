export pr, Cudd

abstract type InferAlgo end
struct Cudd <: InferAlgo end

"Compute probability of a Dice.jl program"
pr(x::Bool) = x ? 1.0 : 0.0
pr(x::Dist) = pr(x, Cudd())

include("cudd.jl")