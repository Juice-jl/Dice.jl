##################################
# CUDD Inference
##################################

export WMC

using DataStructures: LinkedList, cons, nil

mutable struct WMC
    c::BDDCompiler
    cache::Dict{CuddNode, Union{AbstractFloat, ADNode}}
    function WMC(c)
        w = new(c, Dict{CuddNode, Union{AbstractFloat, ADNode}}())
        w.cache[constant(w.c.mgr, true)] = log(one(Float64))
        w.cache[constant(w.c.mgr, false)] = log(zero(Float64))
        w
    end
end

logprob(w::WMC, x::AnyBool) = logprob(w, compile(w.c, x))

function node_logprob(level_pr, hi_logpr, lo_logpr)
    a = log(level_pr) + hi_logpr
    b = log(1.0-level_pr) + lo_logpr
    # Better-accuracy version of log(exp(a) + exp(b))
    if isinf(a)
        b
    elseif isinf(b)
        a
    else
        max(a,b) + log1p(exp(-abs(a-b)))
    end
end

function logprob(w::WMC, x::CuddNode)
    get!(w.cache, x) do
        f = w.c.level_to_flip[level(x)]
        node_logprob(f.prob, logprob(w, high(x)), logprob(w, low(x)))
    end
end
