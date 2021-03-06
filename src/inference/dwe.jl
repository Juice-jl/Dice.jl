# Dist with error

export DWE

struct DWE{T} <: Any where T <: Dist
    d::T
    err::DistBool
end

DWE(d::T) where T <: Dist = DWE{T}(d, DistBool(false))

to_dist(d::DWE) = d

function replace_helper(d::DWE, mapping)
    DWE(replace(d.d, mapping), replace(d.err, mapping))
end

# Check if output is a Tuple{Dist, DistBool}; include DistBool in error if so
function dwe_wrap(x, err)
    if x isa Tuple{Dist, DistBool}
        DWE(x[1], x[2] | err)
    else
        DWE(x, err)
    end
end

# Three-operand operations
prob_setindex(x::DWE, y::DWE, z::DWE) = dwe_wrap(prob_setindex(x.d, y.d, z.d), x.err | y.err | z.err)

function ifelse(cond::DWE, then::DWE, elze::DWE)
    d = ifelse(cond.d, then.d, elze.d)
    err = cond.err | ifelse(cond.d, then.err, elze.err)
    DWE(d, err)
end

# Allow promotion to DWE
for op in (:ifelse, :prob_setindex)
    eval(quote
        $op(x::DWE, y::Dist, z::Dist) = $op(x, DWE(y), DWE(z))
        $op(x::Dist, y::DWE, z::Dist) = $op(DWE(x), y, DWE(z))
        $op(x::Dist, y::Dist, z::DWE) = $op(DWE(x), DWE(y), z)
        $op(x::DWE, y::DWE, z::Dist) = $op(x, y, DWE(z))
        $op(x::DWE, y::Dist, z::DWE) = $op(x, DWE(y), z)
        $op(x::Dist, y::DWE, z::DWE) = $op(DWE(x), y, z)
    end)
end

# Binary operations
for op in (:(Base.:+), :(Base.:-), :(Base.:*), :(Base.:/), :(Base.:%),
        :(Base.:&), :(Base.:|), :(Base.:>), :(Base.:<), :(Base.getindex), 
        :prob_equals, :prob_append, :prob_extend, :prob_append_child, 
        :prob_extend_children, :prob_startswith)
    eval(quote
        $op(x::DWE, y::DWE) = dwe_wrap($op(x.d, y.d), x.err | y.err)
        # Allow promotion to DWE
        $op(x::Dist, y::DWE) = $op(DWE(x), y)
        $op(x::DWE, y::Dist) = $op(x, DWE(y))
    end)
end

# Unary operations
Base.:!(x::DWE) = dwe_wrap(!x.d, x.err)
leaves(x::DWE) = dwe_wrap(leaves(x.d), x.err)

bools(x::DWE) = vcat(bools(x.d), bools(x.err))
