export value, compute, differentiate, Valuation, Derivs, ADComputer

using DirectedAcyclicGraphs
using DataStructures: DefaultDict

Valuation = Dict{Var, ADNodeCompatible}
Derivs = Dict{ADNode, ADNodeCompatible}

mutable struct ADComputer
    vals::Dict{ADNode, ADNodeCompatible}
    function ADComputer(var_vals::Valuation)
        new(merge(Dict{ADNode, ADNodeCompatible}(), var_vals))
    end
end

function compute(a::ADComputer, root::ADNode)
    foldup(root, compute_leaf, compute_inner, ADNodeCompatible, a.vals)
end

compute(var_vals::Valuation, root::ADNode) = compute(var_vals, [root])[root]
function compute(var_vals::Valuation, roots)
    a = ADComputer(var_vals)
    for root in roots
        compute(a, root)
    end
    a.vals
end

function differentiate(var_vals::Valuation, root_derivs::Derivs)
    vals = compute(var_vals, keys(root_derivs))
    derivs = merge(Dict{ADNode, ADNodeCompatible}(), root_derivs)
    foreach_down(keys(root_derivs)) do n
        haskey(derivs, n) && backward(n, vals, derivs)
    end
    vals, derivs
end

# Extending DirectedAcyclicGraphs.jl
function foreach_down(f::Function, roots)
    seen = Set()
    rev_topo = []
    function visit(n)
        n ∈ seen && return
        push!(seen, n)
        if isinner(n)
            for child in children(n)
                visit(child)
            end
        end
        push!(rev_topo, n)
    end
    for root in roots
        visit(root)
    end

    for n in Iterators.reverse(rev_topo)
        f(n)
    end
end
