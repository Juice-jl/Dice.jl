# Demo of using BDD MLE to learn flip probs for a sorted nat list of uniform length.

using Dice

# Return a List
function gen_sorted_list(size, lo, hi)
    size == 0 && return DistNil(DistUInt32)
    
    # The flips used in the uniform aren't tracked via flip_for, so we
    # don't learn their probabilities (this is on purpose - we could).
    @dice_ite if flip_for(size)
        DistNil(DistUInt32)
    else
        # Try changing the parameter to flip_for to a constant, which would force
        # all sizes to use the same probability.
        x = unif(lo, hi)
        DistCons(x, gen_sorted_list(size-1, x, hi))
    end

end

# Top-level size/fuel. For gen_list, this is the max length.
INIT_SIZE = 5

# Dataset over the desired property to match. Below is a uniform distribution
# over sizes.
DATASET = [DistUInt32(x) for x in 0:INIT_SIZE]

# Use Dice to build computation graph
list = gen_sorted_list(
    INIT_SIZE,
    DistUInt32(1),
    DistUInt32(INIT_SIZE),
)
list_len = length(list)

println("Distribution before training:")
display(pr(list_len))
println()

bools_to_maximize = [prob_equals(list_len, x) for x in DATASET]
train_group_probs!(bools_to_maximize, 1000, 0.3)  # epochs and lr

# Done!
println("Learned flip probability for each size:")
display(get_group_probs())
println()

println("Distribution over lengths after training:")
display(pr(list_len))
println()

println("A few sampled lists:")
for _ in 1:3
    print_tree(sample(list))
    println()
end

#==
Distribution before training:
   0 => 0.49999999999999994
   1 => 0.24999999999999972
   2 => 0.12499999999999986
   3 => 0.062499999999999924
   4 => 0.03124999999999996
   5 => 0.03124999999999996

Learned flip probability for each size:
   1 => 0.5
   2 => 0.33333333333333337
   3 => 0.25000000000000006
   4 => 0.19999999999999996
   5 => 0.16666666666666663

Distribution over lengths after training:
   2 => 0.1666666666666666
   3 => 0.1666666666666666
   0 => 0.16666666666666646
   1 => 0.16666666666666646
   4 => 0.16666666666666646
   5 => 0.16666666666666646

A few sampled lists:
Cons
├── 1
└── Cons
    ├── 2
    └── Cons
        ├── 3
        └── Cons
            ├── 3
            └── Cons
                ├── 4
                └── Nil

Cons
├── 3
└── Cons
    ├── 3
    └── Cons
        ├── 3
        └── Cons
            ├── 3
            └── Cons
                ├── 5
                └── Nil

Cons
├── 3
└── Cons
    ├── 5
    └── Nil
==#
