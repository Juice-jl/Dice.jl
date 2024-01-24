export Cudd, CuddDebugInfo

using DataStructures: OrderedDict

struct CuddDebugInfo
    num_nodes::Integer
end

struct Cudd <: InferAlgo
    reordering_type::CUDD.Cudd_ReorderingType
    debug_info_ref::Union{Ref{CuddDebugInfo}, Nothing}
end

function Cudd(;reordering_type=CUDD.CUDD_REORDER_NONE, debug_info_ref=nothing)
    Cudd(reordering_type, debug_info_ref)
end

default_infer_algo() = Cudd()

function get_world_probs(w::WMC, query::JointQuery, evidence::AnyBool)
    # compile BDD for evidence
    evid_bdd = compile(w.c, evidence)
    evid_logp = logprob(w, evid_bdd)

    # get values of adnodes
    vals = Dict{ADNode, ADNodeCompatible}()

    # TODO should query bits be made unique to save time?    
    states = Pair{LinkedList, Float64}[]

    function rec(context::CuddNode, state, rembits)
        issat(w.c.mgr, context) || return
        if isempty(rembits)
            p = exp(logprob(w, context) - evid_logp)
            push!(states, state => p)
        else
            head = rembits[1]
            tail = @view rembits[2:end]
            ifbdd, elsebdd = split(w.c, context, head)
            rec(ifbdd, cons(head => true, state), tail)
            rec(elsebdd, cons(head => false, state), tail)
        end
    end

    if issat(w.c.mgr, evid_bdd) && isempty(query.bits) 
        push!(states, nil() => 1.0)
    else
        rec(evid_bdd, nil(), query.bits)
    end
    @assert !isempty(states) "Cannot find any possible worlds"
    [(Dict(state), p) for (state, p) in states]
end


function pr_impl(cudd::Cudd, evidence, queries::Vector{JointQuery}, errors, dots)
    w = WMC(
        BDDCompiler(Iterators.flatten((
            Iterators.flatten(query.bits for query in queries),
            (err[1] for err in errors),
            [evidence],
            Iterators.flatten(xs for (xs, filename) in dots),
        )))
    )

    enable_reordering(w.c, cudd.reordering_type)

    # compile BDDs and infer probability of all errors
    prob_errors = ProbError[]
    for (cond, err) in errors
        logp = logprob(w, cond)
        isinf(logp) || push!(prob_errors, (exp(logp), err))
    end
    isempty(prob_errors) || throw(ProbException(prob_errors))

    for (xs, filename) in dots
        xs = [compile(w.c, x) for x in xs]
        dump_dot(mgr, xs, filename)
    end

    # compile BDDs and infer probability for all queries
    results = [get_world_probs(w, q, evidence) for q in queries]
    
    
    if !isnothing(cudd.debug_info_ref)
        node_count = num_bdd_nodes(w.c.mgr, [w.c.cache[root] for root in w.c.roots])
        cudd.debug_info_ref[] = CuddDebugInfo(node_count)
    end

    for nup in values(w.c.num_uncompiled_parents)
        @assert nup == 0 "Dereferences are likely suboptimal because num_uncompiled_parents was initialized improperly."
    end

    results
end
