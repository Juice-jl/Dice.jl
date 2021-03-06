import IfElse: ifelse
using DirectedAcyclicGraphs
import DirectedAcyclicGraphs: children, NodeType, DAG, Inner, Leaf

export Dist, DistBool, prob_equals, infer_bool, ifelse, flip, bools, dist_to_mgr_and_compiler, flips_left_to_right, flips_by_instantiation_order, flips_by_deepest_depth, flips_by_shallowest_depth, flips_by_freq, clear_flips, flip_probs, Flip, children, hoist!, to_dist, num_flips, compgraph_size

export DistAnd, DistOr, DistNot, DistIte, DistTrue, DistFalse, DistBools

"A probability distribution over values of type `T`"
abstract type Dist{T} <: DAG end

"The global context of a dice program analysis"
abstract type DiceManager end

abstract type DistBool <: Dist{Bool} end

mutable struct DistAnd <: DistBool
    x::DistBool
    y::DistBool
end

mutable struct DistOr <: DistBool
    x::DistBool
    y::DistBool
end

mutable struct DistNot <: DistBool
    x::DistBool
end

mutable struct DistEquals <: DistBool
    x::DistBool
    y::DistBool
end

mutable struct DistIte <: DistBool
    c::DistBool
    t::DistBool
    e::DistBool
end

struct DistTrue <: DistBool end

struct DistFalse <: DistBool end

mutable struct Flip <: DistBool
    id::Int
end

# Wrap Vector{Bool} on behalf of DAG
mutable struct DistBools <: Dist{Vector{Bool}}
    bools::Vector{DistBool}
end

DistBools(x::Dist) = DistBools(bools(x))

flip_next_id = 1
flip_probs = Dict{Int, AbstractFloat}()
# for debugging purposes... GC is ideal
function clear_flips() 
    global flip_next_id = 1
    global flip_probs = Dict{Int, AbstractFloat}()
end

function flip(p::Real)
    if iszero(p)
        return DistFalse()
    elseif isone(p)
        return DistTrue()
    end
    f = Flip(flip_next_id)
    global flip_probs[flip_next_id] = p
    global flip_next_id += 1
    f
end

DistBool(b::Bool) =
    if b DistTrue() else DistFalse() end

DistBool(p::Real) =
    if iszero(p)
        DistFalse()
    elseif isone(p)
        DistTrue()
    else
        flip(p)
    end

to_dist(b::Bool) = DistBool(b)

to_dist(d::Dist) = d

Base.show(io::IO, x::DistBool) = print(io, "$(typeof(x))@$(hash(x)?? 10000000000000)")

# Abuse of method overloading to keep computation graph flat
Base.:&(x::DistFalse, y::DistFalse) = DistFalse()
Base.:&(x::DistFalse, y::DistTrue) = DistFalse()
Base.:&(x::DistTrue, y::DistFalse) = DistFalse()
Base.:&(x::DistTrue, y::DistTrue) = DistTrue()
Base.:&(x::DistFalse, y::DistBool) = DistFalse()
Base.:&(x::DistBool, y::DistFalse) = DistFalse()
Base.:&(x::DistTrue, y::DistBool) = y
Base.:&(x::DistBool, y::DistTrue) = x
Base.:&(x::DistBool, y::DistBool) = 
    if x isa Flip && y isa Flip && x.id == y.id
        x
    else
        DistAnd(x, y)
    end

Base.:&(x::DistBool, y::Bool) = x & DistBool(y)
Base.:&(x::Bool, y::DistBool) = DistBool(x) & y

Base.:|(x::DistFalse, y::DistFalse) = DistFalse()
Base.:|(x::DistFalse, y::DistTrue) = DistTrue()
Base.:|(x::DistTrue, y::DistFalse) = DistTrue()
Base.:|(x::DistTrue, y::DistTrue) = DistTrue()
Base.:|(x::DistFalse, y::DistBool) = y
Base.:|(x::DistBool, y::DistFalse) = x
Base.:|(x::DistTrue, y::DistBool) = DistTrue()
Base.:|(x::DistBool, y::DistTrue) = DistTrue()
Base.:|(x::DistBool, y::DistBool) = 
    if x isa Flip && y isa Flip && x.id == y.id
        x
    else
        DistOr(x, y)
    end

Base.:|(x::DistBool, y::Bool) = x | DistBool(y)
Base.:|(x::Bool, y::DistBool) = DistBool(x) | y

Base.:!(x::DistFalse) = DistTrue()
Base.:!(x::DistTrue) = DistFalse()
Base.:!(x::DistNot) = x.x
Base.:!(x::DistBool) = DistNot(x)

prob_equals(x::DistFalse, y::DistFalse) = DistTrue()
prob_equals(x::DistFalse, y::DistTrue) = DistFalse()
prob_equals(x::DistTrue, y::DistFalse) = DistFalse()
prob_equals(x::DistTrue, y::DistTrue) = DistTrue()
prob_equals(x::DistFalse, y::DistBool) = !y
prob_equals(x::DistBool, y::DistFalse) = !x
prob_equals(x::DistTrue, y::DistBool) = y
prob_equals(x::DistBool, y::DistTrue) = x
prob_equals(x::DistBool, y::DistBool) =
    if x isa Flip && y isa Flip && x.id == y.id
        DistTrue()
    else
        DistEquals(x, y)
    end

prob_equals(x::DistBool, y::Bool) = prob_equals(x, DistBool(y))
prob_equals(x::Bool, y::DistBool) = prob_equals(DistBool(x), y)

ifelse(cond::DistFalse, then::DistFalse, elze::DistFalse) = DistFalse()
ifelse(cond::DistFalse, then::DistFalse, elze::DistTrue) = DistTrue()
ifelse(cond::DistFalse, then::DistTrue, elze::DistFalse) = DistFalse()
ifelse(cond::DistFalse, then::DistTrue, elze::DistTrue) = DistTrue()

ifelse(cond::DistTrue, then::DistFalse, elze::DistFalse) = DistFalse()
ifelse(cond::DistTrue, then::DistFalse, elze::DistTrue) = DistFalse()
ifelse(cond::DistTrue, then::DistTrue, elze::DistFalse) = DistTrue()
ifelse(cond::DistTrue, then::DistTrue, elze::DistTrue) = DistTrue()

ifelse(cond::DistTrue, then::DistBool, elze::DistBool) = then
ifelse(cond::DistFalse, then::DistBool, elze::DistBool) = elze
ifelse(cond::DistBool, then::DistFalse, elze::DistFalse) = DistFalse()
ifelse(cond::DistBool, then::DistTrue, elze::DistFalse) = cond
ifelse(cond::DistBool, then::DistFalse, elze::DistTrue) = !cond
ifelse(cond::DistBool, then::DistTrue, elze::DistTrue) = DistTrue()

ifelse(cond::DistBool, then::DistBool, elze::DistBool) = DistIte(cond, then, elze)
ifelse(cond::DistBool, x::Bool, y::DistBool) = ifelse(cond, DistBool(x), y)
ifelse(cond::DistBool, x::DistBool, y::Bool) = ifelse(cond, x, DistBool(y))
ifelse(cond::DistBool, x::Bool, y::Bool) = ifelse(cond, DistBool(x), DistBool(y))

ifelse(cond::DistBool, then::Tuple, elze::Tuple) = tuple([ifelse(cond, t, e) for (t, e) in zip(then, elze)]...)

bools(b::DistBool) = [b]

NodeType(::Type{<:DistAnd}) = Inner()
NodeType(::Type{<:DistOr}) = Inner()
NodeType(::Type{<:DistNot}) = Inner()
NodeType(::Type{<:DistEquals}) = Inner()
NodeType(::Type{<:DistIte}) = Inner()
NodeType(::Type{<:DistBools}) = Inner()
NodeType(::Type{<:DistTrue}) = Leaf()
NodeType(::Type{<:DistFalse}) = Leaf()
NodeType(::Type{<:Flip}) = Leaf()

children(x::DistAnd) = [x.x, x.y]
children(x::DistOr) = [x.x, x.y]
children(x::DistNot) = [x.x]
children(x::DistEquals) = [x.x, x.y]
children(x::DistIte) = [x.c, x.t, x.e]
children(::DistTrue) = []
children(::DistFalse) = []
children(x::DistBools) = x.bools
children(::Flip) = []

compgraph_size(x) = DirectedAcyclicGraphs.num_nodes(DistBools(x))

function flips_left_to_right(x)::Vector{Int}
    flip_ids_set = Set{Int}()
    flip_ids = Vector{Int}()
    foreach(DistBools(x)) do y
        if y isa Flip && y.id ??? flip_ids_set
            push!(flip_ids_set, y.id)
            push!(flip_ids, y.id)
        end
    end
    flip_ids
end

function flips_by_instantiation_order(x)::Vector{Int}
    flip_ids_set = Set{Int}()
    foreach(DistBools(x)) do y
        if y isa Flip
            push!(flip_ids_set, y.id)
        end
    end
    sort(collect(flip_ids_set))
end

# TODO: this is wrong because foreach visits each node once...
function flips_by_freq(x)::Vector{Int}
    flip_id_ctr = Dict{Int, Int}()
    foreach(x) do y
        if y isa Flip
            flip_id_ctr[y.id] = get(flip_id_ctr, y.id, 0) + 1
        end
    end
    [kv[1] for kv in sort(collect(flip_id_ctr), by=kv->(kv[2], -kv[1]))]
end

function flips_by_deepest_depth(x)::Vector{Int}
    # TODO: Make faster by rewriting with DirectedAcyclicGraphs
    println("WARNING: flips_by_deepest_depth will be very slow as it revisits old nodes.")
    flip_id_to_depth = Dict{Int, Int}()
    flips_by_deepest_depth_helper(x, 0, flip_id_to_depth)
    [kv[1] for kv in sort(collect(flip_id_to_depth), by=kv->(kv[2], -kv[1]))]
end

function flips_by_deepest_depth_helper(x, depth::Int, flip_id_to_depth::Dict{Int, Int})
    if x isa Flip
        if !haskey(flip_id_to_depth, x.id) || depth > flip_id_to_depth[x.id]
            flip_id_to_depth[x.id] = depth
        end
    else
        for child in children(x)
            flips_by_deepest_depth_helper(child, depth + 1, flip_id_to_depth)
        end
    end
end

function flips_by_shallowest_depth(x)::Vector{Int}
    # TODO: Make faster by rewriting with DirectedAcyclicGraphs
    println("WARNING: flips_by_shallowest_depth will be very slow as it revisits old nodes.")
    flip_id_to_depth = Dict{Int, Int}()
    flips_by_shallowest_depth_helper(x, 0, flip_id_to_depth)
    [kv[1] for kv in sort(collect(flip_id_to_depth), by=kv->(kv[2], -kv[1]))]
end

function flips_by_shallowest_depth_helper(x, depth::Int, flip_id_to_depth::Dict{Int, Int})
    if x isa Flip
        if !haskey(flip_id_to_depth, x.id) || depth < flip_id_to_depth[x.id]
            flip_id_to_depth[x.id] = depth
        end
    else
        for child in children(x)
            flips_by_shallowest_depth_helper(child, depth + 1, flip_id_to_depth)
        end
    end
end


function gather_constraints(root)
    flip_to_constraints = Dict{Int, Set{Int}}()
    function helper(d, constraints)
        if d isa Flip
            if haskey(flip_to_constraints, d.id)
                intersect!(flip_to_constraints[d.id], constraints)
            else
                flip_to_constraints[d.id] = constraints
            end
        elseif d isa DistIte
            then_and_constraints, then_or_constraints = collect_and_or(d.c)
            else_and_constraints = Set([-id for id in then_or_constraints])
            helper(d.t, then_and_constraints)
            helper(d.e, else_and_constraints)
        else
            for child in children(d)
                helper(child, constraints)
            end
        end
    end

    helper(root, Set())
    flip_to_constraints
end

function probs_and_constraints_to_remapping(flip_probs, flip_to_constraints)
    prob_to_flips = Dict{AbstractFloat, Any}()
    for (flip, p) in flip_probs
        if !haskey(prob_to_flips, p)
            prob_to_flips[p] = Vector()
        end
        push!(prob_to_flips[p], flip)
    end

    flip_remapping = Dict{Int, Int}()

    for (_, flips) in prob_to_flips
        literal_to_flip = Dict{Int, Int}()
        for flip in flips
            if haskey(flip_to_constraints, flip)
                for lit in flip_to_constraints[flip]
                    lit = get(flip_remapping, abs(lit), abs(lit)) * sign(lit)
                    if haskey(literal_to_flip, -lit)
                        parent = literal_to_flip[-lit]
                        flip_remapping[flip] = parent
                    else
                        literal_to_flip[lit] = flip  # note: this overrides
                    end
                end
            end
        end
    end
    flip_remapping
end

# Returns (what must hold if d, what mustn't hold if !d)
function collect_and_or(d::DistBool)
    if d isa Flip
        Set([d.id]), Set([d.id])  # literal can be AND or OR
    elseif d isa DistTrue
        Set(), Set() #nothing  # do not tighten AND constraints, kill OR constraints
    elseif d isa DistFalse
        Set(), Set()  # do not tighten either set of constraints
    elseif d isa DistAnd
        # todo: optimize by only collecting AND
        x_and, x_or = collect_and_or(d.x)
        y_and, y_or = collect_and_or(d.y)
        union(x_and, y_and), Set() # nothing
    elseif d isa DistOr
        # todo: optimize by only collecting OR
        x_and, x_or = collect_and_or(d.x)
        y_and, y_or = collect_and_or(d.y)
        Set(), union(x_or, y_or)
    elseif d isa DistNot
        and, or = collect_and_or(d.x)
        Set(-x for x in or), Set(-x for x in and)  # De Morgan
    elseif d isa DistIte
        t_and, t_or = collect_and_or(d.t)
        e_and, e_or = collect_and_or(d.e)
        intersect(t_and, e_and), intersect(t_or, e_or)
    else
        println("Unimplemented collect_and_or case: $(typeof(d))")  # if only this were OCaml
        Set(), Set()  # kill known constraints
    end
end

replace_cache = Set()
function replace_flips!(d, remapping)
    key = (objectid(d), hash(remapping))
    if key in replace_cache
        return
    end
    push!(replace_cache, key)
    if d isa Flip
        if haskey(remapping, d.id)
            d.id = remapping[d.id]
        end
    else
        for child in children(d)
            replace_flips!(child, remapping)
        end
    end
end

function hoist!(d)
    flip_to_constraints = gather_constraints(bools(d))
    remapping = probs_and_constraints_to_remapping(flip_probs, flip_to_constraints)
    replace_flips!(bools(d), remapping)
end

function num_flips(bits::AbstractVector{T}) where T <: DistBool
    length(flips_left_to_right(bits))
end

function num_flips(d)
    num_flips(bools(d))
end

# Lock in the flip order for a dist
# Returns mgr and function to compile computation graph to BDD
function dist_to_mgr_and_compiler(x; flip_order=nothing, flip_order_reverse=false)
    if flip_order === nothing
        flip_order = flips_by_instantiation_order
    end
    flips_ordered = flip_order(bools(x))
    if flip_order_reverse
        reverse!(flips_ordered)
    end
    
    mgr = CuddMgr()
    flip_vars = Dict()
    for flip_id in flips_ordered
        flip_vars[flip_id] = new_var(mgr, flip_probs[flip_id])
    end
    # TODO: do we need to memoize?
    cache = Dict{UInt64, Ptr{Nothing}}()
    function to_bdd_mem(x)
        oid = objectid(x)
        if !haskey(cache, oid)
            cache[oid] = to_bdd(x)
        end
        cache[oid]
    end
    function to_bdd(x::DistAnd) conjoin(mgr, to_bdd_mem(x.x), to_bdd_mem(x.y)) end
    function to_bdd(x::DistOr) disjoin(mgr, to_bdd_mem(x.x), to_bdd_mem(x.y)) end
    function to_bdd(x::DistNot) negate(mgr, to_bdd_mem(x.x)) end
    function to_bdd(x::DistEquals) biconditional(mgr, to_bdd_mem(x.x), to_bdd_mem(x.y)) end
    function to_bdd(x::DistIte) ite(mgr, to_bdd_mem(x.c), to_bdd_mem(x.t), to_bdd_mem(x.e)) end
    to_bdd(_::DistTrue) = constant(mgr, true)
    to_bdd(_::DistFalse) = constant(mgr, false)
    to_bdd(x::Flip) = flip_vars[x.id]  # If KeyError - make sure all flips are contained in the dist passed in to dist_to_mgr_and_compiler
    return mgr, to_bdd_mem
end

function infer_bool(d::DistBool; observation::DistBool=DistTrue())
    mgr, to_bdd = dist_to_mgr_and_compiler([d, observation])
    infer_bool(mgr, to_bdd(d & observation))/infer_bool(mgr, to_bdd(observation))
end

function infer_bool(inferer, d::DistBool; observation::DistBool=DistTrue())
    inferer(d & observation)/inferer(observation)
end
