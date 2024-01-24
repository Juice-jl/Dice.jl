# Demo of using BDD MLE to learn flip probs for nat list of uniform length.

using Dice

function gen_list(size)
    size == 0 && return DistNil(DistUInt32)

    # Try changing the parameter to flip_for to a constant, which would force
    # all sizes to use the same probability.
    @dice_ite if flip_for(size)
        DistNil(DistUInt32)
    else
        # The flips used in the uniform aren't tracked via flip_for, so we
        # don't learn their probabilities (this is on purpose - we could).
        DistCons(uniform(DistUInt32, 0, 10), gen_list(size-1))
    end
end

# Top-level size/fuel. For gen_list, this is the max length.
INIT_SIZE = 5

# Dataset over the desired property to match. Below is a uniform distribution
# over sizes.
DATASET = [DistUInt32(x) for x in 0:INIT_SIZE]

# Use Dice to build computation graph
list = gen_list(INIT_SIZE)
list_len = length(list)

println("Distribution before training:")
display(pr(list_len))
println()

bools_to_maximize = AnyBool[prob_equals(list_len, x) for x in DATASET]
train_group_probs!(bools_to_maximize, 1000, 0.3)  # epochs and lr

# Done!
println("Learned flip probability for each size:")
display(get_group_probs())
println()

println("Distribution over lengths after training:")
display(pr(list_len))

#==
Distribution before training:
   0 => 0.5
   1 => 0.25
   2 => 0.12500000000000003
   3 => 0.0625
   4 => 0.03125
   5 => 0.03125

Learned flip probability for each size:
   1 => 0.5
   2 => 0.33333333333333337
   3 => 0.25000000000000006
   4 => 0.20000000000000007
   5 => 0.16666666666666669

Distribution over lengths after training:
   1 => 0.1666666666666667
   0 => 0.16666666666666669
   2 => 0.16666666666666669
   3 => 0.16666666666666669
   4 => 0.1666666666666666
   5 => 0.1666666666666666
==#
