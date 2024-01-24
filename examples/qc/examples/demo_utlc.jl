# Using BDD MLE to learn flip probs for closed UTLC exprs of uniform AST depth

using Dice
include("lib/dist_utlc.jl")     # DistVar, DistApp, DistAbs, utlc_str

function gen_name()
    @dice_ite if flip(1/3)
        DistString("a")
    elseif flip(1/2)
        DistString("b")
    else
        DistString("c")
    end
end

# Return ast, evid pair
function gen_utlc(size, in_scope)
    # Generate Var arg
    name = choice(in_scope)

    # Fuel check
    size == 0 && return DistVar(name)

    @dice_ite if flip_for(size) & (in_scope.len > DistUInt32(0))
        DistVar(name)
    # Fix weight between Abs and App. We must also always choose Abs if
    # size=1 and the scope is empty so far.
    elseif flip(2/3) | (size==1 & prob_equals(in_scope.len, DistUInt32(0)))
        fresh = gen_name()
        DistAbs(
            fresh,
            gen_utlc(size-1, prob_append(in_scope, fresh))
        )
    else
        DistApp(gen_utlc(size-1, in_scope), gen_utlc(size-1, in_scope))
    end
end

# Top-level size/fuel. For gen_bst, this is the max depth.
INIT_SIZE = 4

# Dataset over the desired property to match. Below is a uniform distribution
# over sizes.
DATASET = [DistUInt32(x) for x in 0:INIT_SIZE]

# Use Dice to build computation graph
e = gen_utlc(INIT_SIZE, DistVector{DistString}())
e_depth = ast_depth(e)

println("Distribution before training:")
display(pr(e_depth))
println()

bools_to_maximize = [prob_equals(e_depth, x) for x in DATASET]
train_group_probs!(bools_to_maximize, 1000, 0.3)  # epochs and lr

# Done!
println("Learned flip probability for each size:")
display(get_group_probs())
println()

println("Distribution over depths after training:")
display(pr(e_depth))
println()

println("A few sampled exprs:")
for _ in 1:10
    expr = sample(e)
    println(utlc_str(expr))
    # println(print_tree(expr))  # concrete AST
end

#==
Distribution before training:
   4 => 0.3670624714220393
   1 => 0.3333333333333333
   2 => 0.1759259259259259
   3 => 0.12367826931870127

Learned flip probability for each size:
   1 => 0.7709384509910406
   2 => 0.5673658539871177
   4 => 0.5
   3 => 0.3749999999999999

Distribution over depths after training:
   2 => 0.2500000000000001
   4 => 0.25000000000000006
   1 => 0.2499999999999999
   3 => 0.24999999999999983

A few sampled exprs:
λc.λb.b b
λb.λa.λc.a b
(λa.a) (λa.a)
λb.(λa.λc.c) (b b)
λb.b
λb.b
(λc.c) (λa.λc.λc.c)
(λb.b) (λb.λc.c b)
λb.b
λa.λb.λb.λb.b
==#
