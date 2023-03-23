
using DirectedAcyclicGraphs

export Dist, isdeterministic

# TODO make DirectedAcyclicGraphs.DAG a trait so we are not thrown off by multiple inheritance here
"A probability distribution over values of type `T`"
abstract type Dist{T}  <: DAG end

"Does the distribution have a deterministic value?"
isdeterministic(x) =
    isempty(tobits(x))

function Base.show(io::IO, x::Dist) 
    if isdeterministic(x)
        print(io, "$(typeof(x))($(frombits(x, nothing)))")
    else
        print(io, "$(typeof(x))@$(hash(x)÷ 10000000000000)")
    end
end

include("bool.jl")
include("misc.jl")
include("integer/uint.jl")
include("integer/int.jl")
include("fixedpoint.jl")
include("char.jl")
include("string.jl")
include("enum.jl")
include("vector.jl")
