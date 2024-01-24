export condprobs, condprob, Cudd, CuddDebugInfo, ProbException, allobservations, JointQuery, 
    returnvalue, expectation, variance, kldivergence, tvdistance, entropy, pr

using DataStructures: DefaultDict, DefaultOrderedDict, OrderedDict

##################################
# Core inference API implemented by backends
##################################

"A choice of probabilistic inference algorithm"
abstract type InferAlgo end

"An error with a probabilistic condition"
const CondError = Tuple{AnyBool, ErrorException}

struct JointQuery
    # use a vector so that order is maintained, might be important for some inference heuristics
    bits::Vector{Dist{Bool}}
    JointQuery(bits) = new(unique!(bits))
end

"""
Compute probabilities of queries, optionally given evidence, 
conditional errors, and a custom inference algorithm.
"""
function pr(queries::Vector{JointQuery}; evidence::AnyBool = true, 
            errors::Vector{CondError} = CondError[],
            dots::Vector{Tuple{Vector{AnyBool}, String}} = Tuple{Vector{AnyBool}, String}[],
            algo::InferAlgo = default_infer_algo()) 
    pr_impl(algo, evidence, queries, errors, dots)
end

function pr(queries::JointQuery...; kwargs...)
    ans = pr(collect(queries); kwargs...)
    length(queries) == 1 ? ans[1] : ans
end

function pr(queries...; kwargs...)
    joint_queries = map(queries) do query
        JointQuery(tobits(query))
    end
    queryworlds = pr(collect(joint_queries); kwargs...)
    ans = map(queries, queryworlds) do query, worlds
        dist = DefaultDict(0.0)
        for (world, p) in worlds
            dist[frombits(query, world)] += p
        end
        DefaultOrderedDict(0., OrderedDict(sort(collect(dist); 
                                by= t -> (-t[2], t[1]))))  # by decreasing probability
    end
    length(queries) == 1 ? ans[1] : ans
end

"""Compute the entropy of a random variable"""
function entropy(p)
    -sum(pr(p)) do (_, prob)
        prob * log(prob)
    end
end

"""Compute the KL-divergence between two random variables"""
function kldivergence(p, q)
    prp = pr(p)
    prq = pr(q)
    sum(prp) do (value, prob)
        prob * (log(prob) - log(prq[value]))
    end
end

"""Compute the total variation distance between two random variables"""
function tvdistance(p, q)
    prp = pr(p)
    prq = pr(q)
    0.5 * sum(keys(prp) ∪ keys(prq)) do value
        abs(prp[value] - prq[value])
    end
end

##################################
# Inference with metadata distributions from DSL
##################################
       
"A distribution computed by a dice program with metadata on observes and errors"
struct MetaDist
    returnvalue
    errors::Vector{CondError}
    observations::Vector{AnyBool}
    dots::Vector
end

returnvalue(x) = x.returnvalue
observations(x) = x.observations
allobservations(x) = reduce(&, observations(x); init=true)

function pr(x::MetaDist; ignore_errors=false, 
            algo::InferAlgo = default_infer_algo())
    evidence = allobservations(x)
    errors = ignore_errors ? CondError[] : x.errors
    pr(returnvalue(x); evidence, errors, x.dots, algo)
end

function expectation(x::MetaDist; ignore_errors=false, 
    algo::InferAlgo = default_infer_algo())
evidence = allobservations(x)
errors = ignore_errors ? CondError[] : x.errors
expectation(returnvalue(x); evidence, errors, algo)
end

function variance(x::MetaDist; ignore_errors=false, 
    algo::InferAlgo = default_infer_algo())
evidence = allobservations(x)
errors = ignore_errors ? CondError[] : x.errors
variance(returnvalue(x); evidence, errors, algo)
end

##################################
# Prbabilistic exception support
##################################

"An error that is thrown with some non-zero probability"
const ProbError = Tuple{Float64, ErrorException}

"A collection of errors that is thrown with some non-zero probability"
struct ProbException <: Exception
    errors::Vector{ProbError}
end

function Base.showerror(io::IO, exc::ProbException)
    print(io, "ProbException: ")
    for (p, err) in exc.errors
        print(io, "  (")
        print(io, err)
        print(io, " with probability $p)")        
    end
end

##################################
# Inference backends
##################################

# Wraps CUDD
# Notable exports:
# - CuddNode
# - low(::CuddNode), high(::CuddNode)
# - level_traversal(::CuddNode)
include("cudd/core.jl")

# Manages context for compiling AnyBools to CuddNode
# Notable exports:
# - BDDCompiler(roots::Vector{AnyBool})
# - compile(::BDDCompiler, AnyBool)
include("cudd/compile.jl")

# Manages context for finding logprobability of a CuddNode, and the gradient of
# a node's logprobability w.r.t. all flip probabilities.
# Notable exports:
# - WMC(::BDDCompiler)
# - logprob(::WMC, ::CuddNode)::Float64
# - grad_logprob(::WMC, ::CuddNode)::Dict{Flip, Float64}
include("cudd/wmc.jl")

# Exposes functionality for inferring the distribution of AnyBool-based
# data structures.
# Notable exports:
# - pr(::Dist, evidence=..., errors=...)
include("pr.jl")

include("sample.jl")