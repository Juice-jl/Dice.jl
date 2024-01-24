abstract type Benchmark end
abstract type GenerationParams{T} end
abstract type LossParams{T} end

abstract type Generation{T} end

##################################
# STLC generation
##################################

abstract type STLC <: Benchmark end

struct STLCGenerationParams <: GenerationParams{STLC}
    param_vars_by_size::Bool
    size::Integer
    ty_size::Integer
    STLCGenerationParams(;param_vars_by_size, size, ty_size) = new(param_vars_by_size, size, ty_size)
end
struct STLCGeneration <: Generation{STLC}
    e::DistI{Opt{DistI{Expr}}}
    constructors_overapproximation::Vector{DistI{Opt{DistI{Expr}}}}
end
function to_subpath(p::STLCGenerationParams)
    [
        "stlc",
        if p.param_vars_by_size "by_sz" else "not_by_sz" end,
        "sz=$(p.size),tysz=$(p.ty_size)",
    ]
end
function generate(p::STLCGenerationParams)
    constructors_overapproximation = []
    function add_ctor(v::DistI{Opt{DistI{Expr}}})
        push!(constructors_overapproximation, v)
        v
    end
    e = gen_expr(
        DistNil(DistI{Typ}),
        gen_type(p.ty_size, p.param_vars_by_size),
        p.size,
        p.ty_size,
        p.param_vars_by_size,
        add_ctor,
    )
    STLCGeneration(e, constructors_overapproximation)
end

##################################
# Approx STLC constructor entropy loss
##################################

struct ApproxSTLCConstructorEntropy <: LossParams{STLC} end
to_subpath(::ApproxSTLCConstructorEntropy) = ["approx_entropy"]
function build_loss(::ApproxSTLCConstructorEntropy, generation::STLCGeneration)
    loss = sum(
        neg_entropy(opt_ctor_to_id(ctor), values(stlc_ctor_to_id), ignore_non_support=true)
        for ctor in generation.constructors_overapproximation
    )
    loss, nothing
end

##################################
# Exact STLC constructor entropy loss
##################################

struct STLCConstructorEntropy <: LossParams{STLC} end
to_subpath(::STLCConstructorEntropy) = ["entropy"]
function build_loss(::STLCConstructorEntropy, generation::STLCGeneration)
    random_term = match(generation.e, [
        "Some" => e -> DistSome(choice(collect_constructors(e))),
        "None" => () -> DistNone(DistInt32),
    ])
    loss = neg_entropy(random_term, Set([DistSome(i) for i in values(stlc_ctor_to_id)]))
    loss, nothing
end

##################################
# STLC "4231" (few apps) loss
##################################

struct STLC4321AppsLoss <: LossParams{STLC} end
to_subpath(::STLC4321AppsLoss) = ["4321apps"]
function build_loss(::STLC4321AppsLoss, generation::STLCGeneration)
    metric = num_apps(generation.e)
    mle_loss([
        BoolToMax(prob_equals(metric, DistUInt32(0)), weight=.4),
        BoolToMax(prob_equals(metric, DistUInt32(1)), weight=.3),
        BoolToMax(prob_equals(metric, DistUInt32(2)), weight=.2),
        BoolToMax(prob_equals(metric, DistUInt32(3)), weight=.1),
    ]), nothing
end

##################################
# BST generation
##################################

abstract type BST <: Benchmark end

struct BSTGenerationParams <: GenerationParams{BST}
    size::Integer
    BSTGenerationParams(; size) = new(size)
end
struct BSTGeneration <: Generation{BST}
    t::DistI{Tree}
    constructors_overapproximation::Vector{DistI{Tree}}
end
function to_subpath(p::BSTGenerationParams)
    [
        "bst",
        "sz=$(p.size)",
    ]
end
function generate(p::BSTGenerationParams)
    constructors_overapproximation = []
    function add_ctor(v::DistI{Tree})
        push!(constructors_overapproximation, v)
        v
    end
    t = gen_tree(
        p.size,
        DistNat(0),
        DistNat(40),
        add_ctor,
    )
    BSTGeneration(t, constructors_overapproximation)
end

##################################
# Approx BST constructor entropy loss
##################################

struct ApproxBSTConstructorEntropy <: LossParams{BST} end
to_subpath(::ApproxBSTConstructorEntropy) = ["approx_entropy"]
function build_loss(::ApproxBSTConstructorEntropy, generation::BSTGeneration)
    loss = sum(
        neg_entropy(ctor_to_id(ctor), values(bst_ctor_to_id), ignore_non_support=true)
        for ctor in generation.constructors_overapproximation
    )
    loss, nothing
end

##################################
# MLE loss
##################################

abstract type Metric{T} end
abstract type TargetDist end

struct MLELossParams{T} <: LossParams{T}
    metric::Metric{T}
    target_dist::TargetDist
    MLELossParams(; metric::Metric{T}, target_dist) where T = new{T}(metric, target_dist)
end
to_subpath(p::MLELossParams) = [name(p.metric), name(p.target_dist)]
function build_loss(loss_params::MLELossParams{STLC}, generation::STLCGeneration)
    metric = compute_metric(loss_params.metric, generation)
    metric_loss(metric, loss_params.target_dist), metric
end

struct NumApps <: Metric{STLC} end
compute_metric(::NumApps, gen::STLCGeneration) = num_apps(gen.e)
name(::NumApps) = "num_apps"

struct TermSize <: Metric{STLC} end
compute_metric(::TermSize, gen::STLCGeneration) = term_size(gen.e)
name(::TermSize) = "term_size"

struct Uniform <: TargetDist end
name(::Uniform) = "uniform"
function metric_loss(metric::Dist, ::Uniform)
    mle_loss([
        BoolToMax(prob_equals(metric, DistUInt32(i)))
        for i in support_mixed(metric)
    ])
end

struct Linear <: TargetDist end
name(::Linear) = "linear"
function metric_loss(metric::Dist, ::Linear)
    mle_loss([
        BoolToMax(prob_equals(metric, DistUInt32(i)), weight=i)
        for i in support_mixed(metric)
    ])
end

##################################
# Mixed loss
##################################

struct MixedLossParams{T} <: LossParams{T}
    weighted_losses::Vector{<:Pair{<:LossParams{T}, <:Real}}
end
function to_subpath(p::MixedLossParams)
    [join(
        [
            "$(join(to_subpath(loss), "_"))_w$(weight)"
            for (loss, weight) in p.weighted_losses
        ],
        "_AND_"
    )]
end
function build_loss(p::MixedLossParams{T}, generation::Generation{T}) where T
    mixed_loss = Dice.Constant(0)
    extras = []
    for (loss, weight) in p.weighted_losses
        loss, extra = build_loss(loss, generation)
        mixed_loss += Dice.Constant(weight) * loss
        push!(extras, extra)
    end
    mixed_loss, extras
end

