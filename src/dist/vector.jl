# Vectors
export DistVector, prob_append, prob_extend, prob_startswith, prob_setindex, prob_getindex, prob_contains, choice, choice_obs

mutable struct DistVector{T} <: Dist{Vector} where T <: Any
    contents::Vector{T}
    len::DistUInt32
end

function DistVector{T}() where T <: Any
    DistVector(Vector{T}(), DistUInt32(0))
end

function DistVector{T}(contents::Vector{T}) where T <: Any
    DistVector(contents, DistUInt32(length(contents)))
end

function DistVector(contents::Vector)
    DistVector(contents, DistUInt32(length(contents)))
end

function prob_equals(x::DistVector, y::DistVector)
    res = prob_equals(x.len, y.len)
    for i = 1:min(length(x.contents), length(y.contents))
        res = res & ((DistUInt32(i) > x.len) | prob_equals(x.contents[i], y.contents[i]))
    end
    res
end

function Base.ifelse(cond::Dist{Bool}, then::DistVector{T}, elze::DistVector{T})::DistVector{T} where T <: Any
    mb = max(length(then.contents), length(elze.contents))
    v = Vector{T}(undef, mb)
    for i = 1:mb
        if i > length(then.contents)
            v[i] = elze.contents[i]
        elseif i > length(elze.contents)
            v[i] = then.contents[i]
        else
            v[i] = ifelse(cond, then.contents[i], elze.contents[i])
        end
    end
    DistVector(v, ifelse(cond, then.len, elze.len))
end

function prob_append(d::DistVector{T}, x::T)::DistVector{T} where T <: Any
    v = Vector{T}(undef, length(d.contents) + 1)
    for i = 1:length(d.contents)
        v[i] = ifelse(prob_equals(d.len, DistUInt32(i-1)), x, d.contents[i])
    end
    v[length(d.contents) + 1] = x
    DistVector(v, d.len + DistUInt32(1))
end

function Base.sort(d::DistVector{T}) where T
    isempty(d.contents) && return d

    # get index of greatest element
    max_i = DistUInt32(1)
    max_val = d.contents[1]
    for i in 1:length(d.contents)
        (max_i, max_val) = ifelse(
            (DistUInt32(i) <= d.len) & (d.contents[i] > max_val),
            (DistUInt32(i), d.contents[i]),
            (max_i, max_val)
        )
    end

    # selection sort
    res_indices = DistVector{DistUInt32}()
    res = DistVector{T}()
    for i in 1:length(d.contents)
        # get min element not yet included
        min_i = max_i
        min_val = max_val
        for j in 1:length(d.contents)
            min_i, min_val = ifelse(
                (!prob_contains(res_indices, DistUInt32(j))) & (d.contents[j] <= min_val) & (DistUInt32(j) <= d.len),
                (DistUInt32(j), d.contents[j]),
                (min_i, min_val)
            )
        end
        res_indices, res = ifelse(
            DistUInt32(i) <= d.len,
            (prob_append(res_indices, min_i), prob_append(res, min_val)),
            (res_indices, res)
        )
    end
    res
end

function prob_contains(d::DistVector{T}, x::T) where T
    found = false
    for (i, y) in enumerate(d.contents)
        found = found | ((DistUInt32(i) <= d.len) & prob_equals(x, y))
    end
    found
end

dummy(::Type{DistString}) = DistString("dummy")
dummy(::Type{DistUInt32}) = DistUInt32(555)
dummy(::Type{DistInt32}) = DistInt32(555)

function prob_getindex(d::DistVector{T}, idx::DistUInt32) where T
    ans = if isempty(d.contents) dummy(T) else d.contents[1] end
    for i in 1:length(d.contents)
        ans = @dice_ite if prob_equals(DistUInt32(i), idx)
            d.contents[i]
        else
            ans
        end
    end
    ans
end


# # Divide-and-conquer getindex
# function prob_getindex(d::DistVector, idx::DistUInt32)
#     # (idx < DistUInt32(1) || idx > d.len) && error("Vector out of bounds access")
#     println(pr(idx))
#     @dice_ite begin
#         function helper(i, v)
#             println("i: $(i) v: $(v)")
#             if v == 4294967200
#                 exit(123)
#             else
#                 nothing
#             end
#             # assert_dice()
#             if i > length(idx.bits)
#                 if v < 1 || v > length(d.contents)
#                     d.contents[1]  # dummy
#                 else
#                     d.contents[v]
#                 end
#             else
#                 if idx.bits[i]
#                     # if i == 2
#                     #     # It's unlikely that the top bit is one... investigate if this happens
#                     #     error("uh oh... look at $(@__FILE__):$(@__LINE__)")
#                     #     nothing
#                     # else
#                     #     nothing
#                     # end
#                     helper(i+1, v+2^(length(idx.bits) - i))
#                 else
#                     helper(i+1, v)
#                 end
#             end
#         end
#         helper(1, 0)
#     end
# end

function prob_setindex(s::DistVector, idx::DistUInt32, c::Any)
    errorcheck() & (idx < DistUInt32(1) || idx > s.len) && error("Vector out of bounds access")
    contents = collect(s.contents)
    for i = 1:length(s.contents)
        contents[i] = ifelse(prob_equals(idx, DistUInt32(i)), c, s.contents[i])
    end
    DistVector(contents, s.len)
end

function prob_setindex(s::DistVector, idx::Int, c::Any)
    errorcheck() & (idx < 1 || DistUInt32(idx) > s.len) && error("Vector out of bounds access")
    contents = collect(s.contents)
    contents[idx] = c
    DistVector(contents, s.len)
end

function prob_extend(s::DistVector{T}, t::DistVector{T}) where T <: Any
    isempty(s.contents) && return t
    @dice_ite begin
        len = s.len + t.len
        contents = Vector{T}(undef, length(s.contents) + length(t.contents))
        for i = 1:length(contents)
            # println(i)
            contents[i] = if DistUInt32(i) <= s.len
                prob_getindex(s, DistUInt32(i))
            else
                if (DistUInt32(i) > s.len) & (DistUInt32(i) <= s.len + t.len)
                    prob_getindex(t, DistUInt32(i) - s.len)
                else
                    s.contents[1] # dummy value 
                end
            end
        end
        DistVector(contents, len)
    end
end

function prob_startswith(u::DistVector, v::DistVector)
    @dice_ite if u.len < v.len
        false
    else
        reduce(
            &,
            [
                (DistUInt32(i) > v.len) | prob_equals(u.contents[i], v.contents[i])
                for i in 1:length(v.contents)
            ]
        )
    end
end

tobits(s::DistVector) =
    vcat(collect(Iterators.flatten(tobits(c) for c in s.contents)), tobits(s.len))

function frombits(s::DistVector, world)
    len = frombits(s.len, world)
    collect(frombits(c, world) for c in s.contents[1:len])
end

function choice_obs(v::DistVector{T})::Tuple{T, AnyBool} where T
    if prob_equals(v.len, DistUInt32(0))
        return dummy(T), true
    end
    i, evid = unif_obs(DistUInt32(1), v.len)
    prob_getindex(v, i), evid
end

function choice(v::DistVector{T})::T where T
    @dice_ite if prob_equals(v.len, DistUInt32(0))
        dummy(T)
    else
        i = unif(DistUInt32(1), v.len)
        prob_getindex(v, i)
    end
end