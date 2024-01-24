# An introduction to using MLE to learn flip probabilities

################################################################################
# Maximizing expression probabilities
################################################################################
using Revise
using Dice

# What value for ? maximizes the probability of the following expression?
#   flip(?) & flip(?) & !flip(?)

# Let's check!
p = add_unit_interval_var!("p")
x = flip(p) & flip(p) & !flip(p)
train_vars!([x])
compute(p)  # ~ 2/3

# What just happened?
# - `add_unit_interval_var!()` registers a value in (0,1) to learn (init. 0.5)
# - `train_vars!(bools)` performs maximum likelihood estimation to train
#   the parameter to maximize the product of the probabilities of the bools

# We can also perform computation on params before using them for flip
# probabilities. For example, `x` could have been equivalently defined as:
#   x = flip(p) & flip(p) & flip(1 - p)

clear_vars!()  # call before adding the params of the next "program"

# (For the curious) What's happening under the hood?
# - TODO: discuss `ADNode`s, `compute`
# - TODO: discuss how `add_unit_interval_var!`` wraps a param in `sigmoid`

# If the flips above can have different groups, each can take on a
# different probability.
a = add_unit_interval_var!("a")
b = add_unit_interval_var!("b")
c = add_unit_interval_var!("c")
x = flip(a) & flip(b) & !flip(c)
train_vars!([x])
compute(a)  # 0.8419880024053406
compute(b)  # 0.8419880024053406
compute(c)  # 0.1580119975946594

# We can also keep training to get closer to 1, 1, 0.
train_vars!([x]; epochs=10000, learning_rate=3.0)
compute(a)  # 0.9999666784828445
compute(b)  # 0.9999666784828445
compute(c)  # 3.332151715556398e-5

clear_vars!()


################################################################################
# Approximating datasets
################################################################################

# Consider the following probabilistic program with holes. What probability
# will cause x to be true 2/3 of the time?
p = add_unit_interval_var!("p")
b = @dice_ite if flip(p) true else flip(0.5) end

# We can match this dataset...
dataset = [true, true, false]

# ...by maximizing these distributions over bools:
bools_to_maximize = [prob_equals(b, x) for x in dataset]
train_vars!(bools_to_maximize; learning_rate=0.1)
compute(p) # ~ 1/3

# We can also check how close we are by performing normal Dice inference.
# Let's back up:
clear_vars!()
p = add_unit_interval_var!("p")
b = @dice_ite if flip(p) true else flip(0.5) end

# By default, unit interval params have a value of 0.5
pr(b)  # true => 0.75

train_vars!([prob_equals(b, x) for x in dataset]; learning_rate=0.1)

# The program is true 2/3 of the time, as desired!
pr(b)  # true  => 0.666667

clear_vars!()
p = add_unit_interval_var!("p")
# As before, we can also constrain multiple flips to have the same probability
#                                  vvvvvvv changed
b = @dice_ite if flip(p) true else flip(p) end
train_vars!([prob_equals(b, x) for x in dataset]; learning_rate=0.1)
compute(p) # 0.42264973081037427
pr(b)  # true => 0.666667

clear_vars!()

# Here's an example for ints. Lets say we build an int whose bits are flips such
# that it has a uniform distribution over 0, 2, 4, ..., 14.
probs = [add_unit_interval_var!("b$(i)") for i in 1:4]
n = DistUInt{4}([flip(prob) for prob in probs])
dataset = [DistUInt{4}(even) for even in 0:2:14]
bools_to_maximize = [prob_equals(n, x) for x in dataset]
train_vars!(bools_to_maximize; learning_rate=0.1)
[compute(prob) for prob in probs] # 0.5, 0.5, 0.5, 0.000626043181613181
clear_vars!()

################################################################################
###################### TODO: update the rest of this file ######################
################################################################################

# ################################################################################
# # Uniform list lengths
# ################################################################################

# # Consider this recursive function which generates lists up to a certain size
# function gen_list(size)
#     size == 0 && return DistNil(DistUInt32)

#     @dice_ite if flip_for("?")
#         DistNil(DistUInt32)
#     else
#         # The flips used in the uniform aren't tracked via flip_for, so we
#         # don't learn their probabilities.
#         DistCons(uniform(DistUInt32, 0, 10), gen_list(size-1))
#     end
# end

# # What if we want to learn a probability for ? such that the list lengths are
# # uniformly distributed from 0 to 5?

# # Note that the distribution of gen_list depends on the initial size passed to
# # the top call. Let's choose to pass in 5 to the top call since we don't want to
# # generate lists larger than that.

# init_size = 5
# dataset = [DistUInt32(x) for x in 0:init_size]
# generated = len(gen_list(init_size))

# # Distribution over lengths before training
# pr(generated)
# #    0 => 0.5
# #    1 => 0.25
# #    2 => 0.12500000000000003
# #    3 => 0.0625
# #    4 => 0.03125
# #    5 => 0.03125

# bools_to_maximize = AnyBool[prob_equals(generated, x) for x in dataset]
# train_group_probs!(bools_to_maximize)
# get_group_probs()
# #   "?" => 0.25

# reset_flips!()

# ################################################################################
# # Uniform list lengths with dynamic groups
# ################################################################################

# # Suppose instead that we can have the flip probability change depend based on
# # the size in the subcall.
# function gen_list(size)
#     size == 0 && return DistNil(DistUInt32)
    
#     #                     vvvvv changed
#     @dice_ite if flip_for(size)
#         DistNil(DistUInt32)
#     else
#         # The flips used in the uniform aren't tracked via flip_for, so we
#         # don't learn their probabilities.
#         DistCons(uniform(DistUInt32, 0, 10), gen_list(size-1))
#     end
# end

# generated = len(gen_list(init_size))
# bools_to_maximize = AnyBool[prob_equals(generated, x) for x in dataset]
# train_group_probs!(bools_to_maximize)
# get_group_probs()
# # 5 => 0.166667
# # 4 => 0.2
# # 2 => 0.333333
# # 3 => 0.25
# # 1 => 0.5

# ################################################################################
# # Including evidence
# ################################################################################

# # So far, we've been passing a list like [x, y, z, ...] to `train_group_probs`.
# # This chooses flip_for probabilities to maximize Pr(x) * Pr(y) * Pr(z)

# # We can also pass a list of tuples to `train_group_probs`, like:
# #   [(x, evid1), (y, evid2), (z, evid3)]
# # This chooses flip_for probabilities to maximize:
# #   Pr(x given evid1) * Pr(y given evid2) * Pr(z given evid3)

# reset_flips!()

# # Lets say we know that x was sampled an unknown dist. over 1, 2, and 3
# x = @dice_ite begin
#     if flip_for("f1")
#         DistUInt32(1) 
#     elseif flip_for("f2")
#         DistUInt32(2)
#     else
#         DistUInt32(3)
#     end
# end

# # Given that x is odd, what flip probabilities cause x to be uniformly distributed?
# is_odd(x::DistUInt32) = prob_equals(x % DistUInt32(2), DistUInt32(1))
# evid = is_odd(x)
# pr(x, evidence=evid)  # 1 => 0.6667, 3 => 0.3333 - not what we want

# train_group_probs!(
#     [
#         (prob_equals(x, DistUInt32(1)), evid),
#         (prob_equals(x, DistUInt32(3)), evid),
#         # Datapoints that indicate x is 2 have no effect under an observation
#         # that removes these worlds
#         (prob_equals(x, DistUInt32(2)), evid),
#         (prob_equals(x, DistUInt32(2)), evid),
#     ]
# )
# get_group_probs()

# pr(x, evidence=evid) # 1 => 0.5, 3 => 0.5

# # More realistic examples for this include demo_sortednatlist.jl and
# # demo_bst.jl.
