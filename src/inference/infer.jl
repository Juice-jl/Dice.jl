export infer, group_infer, infer_with_observation, @Pr

struct InferenceResult
    dist::Dict
    error_p::AbstractFloat
end

# Allow tuple destructuring
Base.iterate(x::InferenceResult) = (x.dist, 1)
Base.iterate(x::InferenceResult, state) = if state == 1 (x.error_p, 2) else nothing end

# Make a few functions go through to dist
for f in (:(Base.length), :(Base.getindex), :(Base.keys), :(Base.values), :(Base.haskey), :(Base.get))
    eval(quote $f(x::InferenceResult, args...) = $f(x.dist, args...) end)
end

function infer(d;
        observation::DistBool=DistTrue(),
        err::DistBool=DistFalse())
    if d isa DWE
        err |= d.err
        d = d.d
    end
    ans = Dict()
    error_p = Ref(0.)
    group_infer(observation, true, 1.0) do observation_met, observe_prior, denom
        if !observation_met return end
        group_infer(err, observe_prior, denom) do error, error_prior, error_p_
            if error
                # Hack to assign out of scope, there must a better way...
                error_p[] = error_p_/denom
                return
            end
            group_infer(d, error_prior, error_p_) do assignment, _, p
                if haskey(ans, assignment)
                    # If this prints, some group_infer implementation is probably inefficent.
                    # println("Warning: Multiple paths to same assignment: $(assignment)")
                    ans[assignment] += p/denom
                else
                    ans[assignment] = p/denom
                end
            end
        end
    end
    InferenceResult(ans, error_p[])
end

macro Pr(code)
    # TODO: use MacroTools or similar to make this more flexible. It would be
    # to support arbitrary expressions, as long as they aren't ambiguous
    # (e.g. multiple |'s ) - or we use || instead of | for boolean or?
    msg = ("@Pr requires the form `@Pr(X)` or `Pr(X | Y)`. As an alternative to"
        * " the macro, consider `infer(X, observation=Y)`.")
    if code isa Symbol
        temp = gensym()
        return quote
            $temp = $(esc(code))
            if $temp isa DistBool
                infer_bool($temp)
            else
                infer($temp)
            end
        end
    else
        @assert (length(code.args) == 3 && code.args[1] == :|) msg
        temp = gensym()
        temp2 = gensym()
        return quote
            $temp = $(esc(code.args[2]))
            $temp2 = $(esc(code.args[3]))
            if $temp isa DistBool
                infer_bool($temp, observation=$temp2)
            else
                infer($temp, observation=$temp2)
            end
        end
    end
end

# Workhorse for group_infer; it's DistBools all the way down
# Params:
#   d is the Dist to infer on
#   prior is a DistBool that must be satisfied
#   prior_p is Pr(prior)
# Behavior:
#   For each satisfying assignment x of d such that prior is true, calls f with:
#   x, a new_prior equivalent to (prior & (d = x)), and Pr(new_prior)
function group_infer(f, d::DistBool, prior, prior_p::Float64)
    new_prior = d & prior
    p = infer_bool(new_prior)
    if p != 0
        f(true, new_prior, p)
    end
    if !(p ≈ prior_p)
        f(false, !d & prior, prior_p - p)
    end
end

# We can infer a vector if we can infer the elements
function group_infer(f, vec::AbstractVector, prior, prior_p::Float64)
    if length(vec) == 0
        f([], prior, prior_p)
        return
    end
    group_infer(vec[1], prior, prior_p) do assignment, new_prior, new_p
        rest = @view vec[2:length(vec)]
        group_infer(rest, new_prior, new_p) do rest_assignment, rest_prior, rest_p
            assignments = vcat([assignment], rest_assignment)  # todo: try linkedlist instead
            f(assignments, rest_prior, rest_p)
        end
    end
end

# Defer to group_infer for vectors (which support @view, useful for efficiency)
function group_infer(f, tup::Tuple, prior, prior_p::Float64)
    group_infer(collect(tup), prior, prior_p) do vec_assignment, new_prior, p
        f(tuple(vec_assignment...), new_prior, p)
    end
end

bools(v::AbstractVector) =
    collect(Iterators.flatten(bools(x) for x in v))

bools(t::Tuple) =
    collect(Iterators.flatten(bools(x) for x in t))