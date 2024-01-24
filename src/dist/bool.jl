export flip, prob_equals, AnyBool, expectation, variance, foreach_node, flip_prob!

##################################
# types, structs, and constructors
##################################

const AnyBool = Union{Dist{Bool}, Bool}

# TODO should become and atomic int when we care about multithreading
global_flip_id::Int64 = one(Int64)

mutable struct Flip <: Dist{Bool}
    const global_id::Int
    prob
    const name
    
    Flip(prob, name) = begin
        if prob isa Real
            @assert !isone(prob) "Use `true` for deterministic flips"
            @assert !iszero(prob) "Use `false` for deterministic flips"
            @assert isnan(prob) || 0 < prob < 1 "Probabilities are between 0 and 1 (or undefined as NaN)"
        end
        global global_flip_id
        new(global_flip_id += 1, prob, name)
    end
end

function Base.show(io::IO, f::Flip)
    p = if f.prob isa AbstractFloat round(f.prob, digits=2) else f.prob end
    if isnothing(f.name)
        print(io, "$(typeof(f))($(f.global_id),$(p))")
    else
        print(io, "$(typeof(f))($(f.global_id),$(p),$(f.name))")
    end
end

"Create a Bernoulli random variable with the given probability (a coin flip)"
function flip(prob = NaN16; name = nothing)
    if prob isa Real
        iszero(prob) && return false
        isone(prob) && return true
    end
    Flip(prob, name)
end

"Set the probability of a flip that has not been assigned a probability yet"
function flip_prob!(f::Flip, prob::Real)
    @assert isnan(f.prob)
    iszero(prob) && return false
    isone(prob) && return true
    @assert isnan(prob) || 0 < prob < 1 "Probabilities are between 0 and 1 (or undefined as NaN), not $prob"
    f.prob = prob
    f
end

abstract type DistBoolOp <: Dist{Bool} end
abstract type DistBoolBinOp <: DistBoolOp end

mutable struct DistAnd <: DistBoolBinOp
    const x::Dist{Bool}
    const y::Dist{Bool}
end

Base.:(&)(x::Dist{Bool}, y::Bool) = y ? x : false
Base.:(&)(x::Bool, y::Dist{Bool}) = y & x
function Base.:(&)(x::Dist{Bool}, y::Dist{Bool}) 
    # TODO figure out if we want `===` or `==` for these optimizations
    x == y && return x
    if (x isa DistBoolBinOp) && (y isa DistBoolBinOp) && 
            (x.x == y.x) && (x.y == y.y) # operand order is mostly fixed by hashcode
        (x isa DistAnd) && return x
        return y
    end
    DistAnd(x,y)
end

mutable struct DistOr <: DistBoolBinOp
    const x::Dist{Bool}
    const y::Dist{Bool}
end

Base.:(|)(x::Dist{Bool}, y::Bool) = y ? true : x
Base.:(|)(x::Bool, y::Dist{Bool}) = y | x
function Base.:(|)(x::Dist{Bool}, y::Dist{Bool}) 
    # TODO figure out if we want `===` or `==` for these optimizations
    x == y && return x
    if (x isa DistBoolBinOp) && (y isa DistBoolBinOp) && 
            (x.x == y.x) && (x.y == y.y) # operand order is mostly fixed by hashcode
        (x isa DistOr) && return x
        return y
    end
    DistOr(x,y)
end

# TODO could be mutable or immutable? see how performance is impacted
struct DistNot <: DistBoolOp
    x::Dist{Bool}
end

# aim to strip negations whenever possible
Base.:(!)(x::Dist{Bool}) = DistNot(x)
Base.:(!)(y::DistNot) = y.x

Base.:(&)(x::Dist{Bool}, y::DistNot) = x == !y ? false : DistAnd(x,y)
Base.:(&)(y::DistNot, x::Dist{Bool}) = x & y
Base.:(&)(x::DistNot, y::DistNot) = !((!x) | (!y))

Base.:(|)(x::Dist{Bool}, y::DistNot) = x == !y ? true : DistOr(x,y)
Base.:(|)(y::DistNot, x::Dist{Bool}) = x | y
Base.:(|)(x::DistNot, y::DistNot) = !((!x) & (!y))

##################################
# inference
##################################

tobits(::Bool) = []
tobits(b::Dist{Bool}) = [b]
frombits(b::Bool, _) = b
frombits(b::Dist{Bool}, world) = world[b]

##################################
# DirectedAcyclicGraphs.jl
##################################

NodeType(::Type{<:DistBoolOp}) = Inner()
NodeType(::Type{<:Flip}) = Leaf()

children(z::DistAnd) = [z.x, z.y]
children(z::DistOr) = [z.x, z.y]
children(z::DistNot) = [z.x]
children(::Flip) = []

##################################
# methods
##################################

prob_equals(x::Bool, y::Bool) = x == y
prob_equals(x::Bool, y::Dist{Bool}) = x ? y : !y
prob_equals(x::Dist{Bool}, y::Bool) = prob_equals(y,x)
prob_equals(x::Dist{Bool}, y::Dist{Bool}) = 
    x == y ? true : (x & y) | (!x & !y)

Base.xor(x::Bool, y::Dist{Bool}) = x ? !y : y
Base.xor(x::Dist{Bool}, y::Bool) = xor(y,x)
Base.xor(x::Dist{Bool}, y::Dist{Bool}) = 
    x == y ? false : (!x | !y) & (x | y)

Base.isless(x::AnyBool, y::AnyBool) = !x & y

Base.:(<=)(x::AnyBool, y::AnyBool) = !isless(y, x)
Base.:(>=)(x::AnyBool, y::AnyBool) = !isless(x, y)

function Base.ifelse(cond::Dist{Bool}, then::AnyBool, elze::AnyBool)
    (then == elze) && return then
    (cond == then) && return cond | elze
    (cond == elze) && return cond & then
    # TODO special case some DistNot branches
    (cond & then) | (!cond & elze)
end

"Test whether at least two of three arguments are true"
atleast_two(x,y,z) = (x & y) | ((x | y) & z)

function foreach_node(f, roots)
    seen = Set{Dist{Bool}}()
    for root in roots
        root isa Bool && continue
        foreach(root) do node
            node ∈ seen && return
            push!(seen, node)
            f(node)
        end
    end
end

##################################
# inference
##################################

"Compute the expected value of a random variable"
expectation(x::AnyBool; kwargs...) = 
    pr(x; kwargs...)[true]

"Compute the variance of a random variable"
function variance(x::AnyBool; kwargs...)
    p = pr(x; kwargs...)
    p[true] * p[false]
end