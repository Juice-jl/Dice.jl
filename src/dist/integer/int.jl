
export DistInt, DistInt8, DistInt16, DistInt32

##################################
# types, structs, and constructors
##################################

"A signed random W-bit integer in two's complement"
struct DistInt{W} <: Dist{Int}
    number::DistUInt{W}
end

function DistInt{W}(b::AbstractVector) where W
    DistInt{W}(DistUInt{W}(b))
end

function DistInt{W}(x::Int) where W
    @assert -(2^(W-1)) <= x < 2^(W-1)
    unsignedx = ifelse(x >= 0, x, x+2^W)
    DistInt{W}(DistUInt{W}(unsignedx))
end

const DistInt8 = DistInt{8}
const DistInt16 = DistInt{16}
const DistInt32= DistInt{32}

function uniform(::Type{DistInt{W}}, n = W) where W
    DistInt{W}(uniform(DistUInt{W}, n).bits)
end

function uniform(::Type{DistInt{W}}, start::Int, stop::Int; kwargs...) where W
    @assert -(2^(W - 1)) <= start < stop <= (2^(W - 1))
    xoff = uniform(DistUInt{W+1}, 0, stop-start; kwargs...)
    x = DistInt{W+1}(xoff) + DistInt{W+1}(start)
    return convert(DistInt{W}, x)
end

# Generates a triangle on positive part of the support
function triangle(t::Type{DistInt{W}}, b::Int) where W
    @assert b < W
    DistInt(triangle(DistUInt{W}, b))
end

function Base.convert(::Type{DistInt{W2}}, x::DistInt{W1}) where W1 where W2
    bits = x.number.bits
    if W1 <= W2
        DistInt{W2}(vcat(fill(bits[1], W2 - W1), bits))
    else
        if errorcheck()
            err = any(b -> !prob_equals(b, bits[W1-W2+1]), bits[1:W1-W2])
            err && error("Cannot convert `DistInt` losslessly from bitwidth $W1 to $W2: $bits")
        end
        DistInt{W2}(bits[W1-W2+1:W1])
    end
end

##################################
# inference
##################################

tobits(x::DistInt) = tobits(x.number)

function frombits(x::DistInt{W}, world) where W
    base = frombits(x.number.bits[1], world) ? -2^(W-1) : 0
    base + frombits(drop_bits(DistUInt{W-1}, x.number), world)
end

function expectation(x::DistInt{W}; kwargs...) where W
    bitprobs = pr(x.number.bits...; kwargs...)
    base = -(2^(W-1)) * bitprobs[1][true]
    base + expectation(bitprobs[2:end])
end

function variance(x::DistInt{W}; kwargs...) where W
    probs = variance_probs(x.number; kwargs...)
    ans = 0
    exponent1 = 1
    for i = 1:W
        ans += exponent1 * (probs[W+1-i, W+1-i] - probs[W+1-i, W+1-i]^2)
        exponent2 = exponent1*2
        for j = i+1:W
            exponent2 = 2*exponent2
            bi = probs[W+1-i, W+1-i]
            bj = probs[W+1-j, W+1-j]
            bibj = probs[W+1-i, W+1-j]
            if j == W
                ans -= exponent2 * (bibj - bi * bj)
            else
                ans += exponent2 * (bibj - bi * bj)
            end
        end
        exponent1 = exponent1*4
    end
    return ans
end

##################################
# methods
##################################

bitwidth(::DistInt{W}) where W = W

function prob_equals(x::DistInt{W}, y::DistInt{W}) where W
    prob_equals(x.number, y.number)
end

function Base.isless(x::DistInt{W}, y::DistInt{W}) where W
    ifelse(x.number.bits[1] & !y.number.bits[1], 
        true, 
        ifelse(!x.number.bits[1] & y.number.bits[1], 
            false, 
            isless(DistUInt{W-1}(x.number.bits[2:W]), DistUInt{W-1}(y.number.bits[2:W]))
        )
    )
end

function Base.ifelse(cond::Dist{Bool}, then::DistInt{W}, elze::DistInt{W}) where W
    DistInt{W}(ifelse(cond, then.number, elze.number))
end

function Base.:(+)(x::DistInt{W}, y::DistInt{W}) where W
    z = convert(DistUInt{W+1}, x.number) + convert(DistUInt{W+1}, y.number)
    if errorcheck()
        overflow = ((!x.number.bits[1] & !y.number.bits[1] & z.bits[2]) | 
                    (x.number.bits[1] & y.number.bits[1] & !z.bits[2]))
        overflow && error("integer overflow or underflow")
    end
    DistInt{W}(drop_bits(DistUInt{W}, z))
end

function Base.:(-)(x::DistInt{W}, y::DistInt{W}) where W
    ans = DistUInt{W+1}(vcat([true], x.number.bits)) - DistUInt{W+1}(vcat([false], y.number.bits))
    borrow = (!x.number.bits[1] & y.number.bits[1] & ans.bits[2]) | (x.number.bits[1] & !y.number.bits[1] & !ans.bits[2])
    errorcheck() & borrow && error("integer overflow or underflow")
    DistInt{W}(ans.bits[2:W+1])
end

function Base.:(*)(x::DistInt{W}, y::DistInt{W}) where W
    p1 = convert(DistInt{2*W}, x).number
    p2 = convert(DistInt{2*W}, y).number
    P = DistUInt{2*W}(0)
    shifted_bits = p1.bits
    for i = 2*W:-1:1
        if (i != 2*W)
            shifted_bits = vcat(shifted_bits[2:2*W], false)
        end
        added = ifelse(p2.bits[i], DistUInt{2*W}(shifted_bits), DistUInt{2*W}(0))
        P = convert(DistUInt{2*W+2}, P) + convert(DistUInt{2*W+2}, added)
        P = drop_bits(DistUInt{2*W}, P)
    end
    P_ans = convert(DistInt{W}, DistInt{2*W}(P))
    P_overflow = DistInt{W}(P.bits[1:W])
    overflow = prob_equals(P_overflow, DistInt{W}(-1)) | iszero(P_overflow)
    errorcheck() & !overflow && error("integer overflow")
    overflow = !iszero(x) & !iszero(y) & ((!xor(p1.bits[W+1], p2.bits[W+1]) & P.bits[W+1]) | (xor(p1.bits[W+1], p2.bits[W+1]) & !P.bits[W+1]))
    errorcheck() & overflow && error("integer overflow")
    return P_ans
end

function Base.:(/)(x::DistInt{W}, y::DistInt{W}) where W
    if errorcheck()
        iszero(y) && error("division by zero")
        overflow = prob_equals(x, DistInt{W}(-2^(W-1))) & prob_equals(y, DistInt{W}(-1))
        errorcheck() & overflow && error("integer overflow")
    end

    xp = @dice_ite if x.number.bits[1]
        DistUInt{W}(1) + ~x.number
    else 
        x.number 
    end
    xp = convert(DistUInt{W+1}, xp)

    yp = @dice_ite if y.number.bits[1] 
        DistUInt{W}(1) + ~y.number
    else 
        y.number 
    end
    yp = convert(DistUInt{W+1}, yp)
    ans = xp / yp

    isneg = xor(x.number.bits[1], y.number.bits[1]) & !iszero(ans)

    ans = @dice_ite if isneg 
        DistUInt{W+1}(1) + DistUInt{W+1}([!ansb for ansb in ans.bits]) 
    else 
        ans 
    end
    ans = DistInt{W}(ans.bits[2:W+1])
    return ans
end

function Base.:(%)(x::DistInt{W}, y::DistInt{W}) where W
    # overflow = prob_equals(x, DistInt{W}(-2^(W-1))) & prob_equals(y, DistInt{W}(-1))
    # overflow && error("integer overflow")

    errorcheck() & iszero(y) && error("division by zero")

    xp = if x.number.bits[1] DistUInt{W}(1) + DistUInt{W}([!xb for xb in x.number.bits]) else x.number end
    xp = convert(DistUInt{W+1}, xp)
    yp = if y.number.bits[1] DistUInt{W}(1) + DistUInt{W}([!yb for yb in y.number.bits]) else y.number end
    yp = convert(DistUInt{W+1}, yp)
    ans = xp % yp

    isneg = x.number.bits[1] & !iszero(ans)

    ans = if isneg DistUInt{W+1}(1) + DistUInt{W+1}([!ansb for ansb in ans.bits]) else ans end
    ans = DistInt{W}(ans.bits[2:W+1])
    return ans
end
