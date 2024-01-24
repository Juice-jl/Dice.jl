export BoolToMax, mle_loss, kl_divergence, neg_entropy

struct BoolToMax
    bool::AnyBool
    evid::AnyBool
    weight::Real
    BoolToMax(b; evidence=true, weight=1) = new(b & evidence, evidence, weight)
end

function mle_loss(bools_to_max::Vector{BoolToMax})
    -1 * sum(bools_to_max) do b
        if b.evid === true
            b.weight * LogPr(b.bool)
        else
            b.weight * (LogPr(b.bool) - LogPr(b.evid))
        end
    end
end
mle_loss(bools::Vector{<:AnyBool}) = mle_loss([BoolToMax(b) for b in bools])

function kl_divergence(p::Dict{<:Any, <:Real}, q::Dist, domain::Set{<:Pair{<:Any, <:Dist}})
    sum(domain) do (x, x_dist)
        logqx = LogPr(prob_equals(q, x_dist))
        p[x] * (log(p[x]) - logqx) 
    end
end

function neg_entropy(p::Dist, domain; ignore_non_support=false)
    sum(domain) do x
        pe = prob_equals(p, x)
        if ignore_non_support && length(support_mixed(pe)) == 1
            Constant(0)
        else
            LogPr(pe) * exp(LogPr(pe))
        end
    end
end
