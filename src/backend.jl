# compilation backend that uses CUDD

using CUDD

struct CuddMgr
    cuddmgr::Ptr{Nothing}
end

function default_manager() 
    cudd_mgr = initialize_cudd()
    Cudd_DisableGarbageCollection(cudd_mgr)
    CuddMgr(cudd_mgr)
end

@inline true_node(mgr::CuddMgr) = 
    Cudd_ReadOne(mgr.cuddmgr)

@inline false_node(mgr::CuddMgr) = 
    Cudd_ReadLogicZero(mgr.cuddmgr)

@inline biconditional(mgr::CuddMgr, x, y) =
    Cudd_bddXnor(mgr.cuddmgr, x, y)

@inline conjoin(mgr::CuddMgr, x, y) =
    Cudd_bddAnd(mgr.cuddmgr, x, y)

@inline disjoin(mgr::CuddMgr, x, y) =
    Cudd_bddOr(mgr.cuddmgr, x, y)

@inline negate(mgr::CuddMgr, x) =
    # workaround until https://github.com/sisl/CUDD.jl/issues/16 is fixed
    biconditional(mgr, x, false_node(mgr))

@inline ite(mgr::CuddMgr, cond, then, elze) =
    Cudd_bddIte(mgr.cuddmgr, cond, then, elze)

@inline issat(mgr::CuddMgr, x) =
    x !== false_node(mgr)

@inline isvalid(mgr::CuddMgr, x) =
    x === true_node(mgr)

@inline new_var(mgr::CuddMgr) =
    Cudd_bddNewVar(mgr.cuddmgr)

@inline num_nodes(mgr::CuddMgr, xs::Vector{<:Ptr}; as_add=true) = begin
    as_add && (xs = map(x -> Cudd_BddToAdd(mgr.cuddmgr, x), xs))
    Cudd_SharingSize(xs, length(xs))
end
        
@inline num_flips(mgr::CuddMgr) =
    Cudd_ReadSize(mgr.cuddmgr)

mutable struct FILE end

@inline dump_dot(mgr::CuddMgr, xs::Vector{<:Ptr}, filename; as_add=true) = begin
    # convert to ADDs in order to properly print terminals
    if as_add
        xs = map(x -> Cudd_BddToAdd(mgr.cuddmgr, x), xs)
    end
    outfile = ccall(:fopen, Ptr{FILE}, (Cstring, Cstring), filename, "w")
    Cudd_DumpDot(mgr.cuddmgr, length(xs), xs, C_NULL, C_NULL, outfile) 
    @assert ccall(:fclose, Cint, (Ptr{FILE},), outfile) == 0
    nothing
end
