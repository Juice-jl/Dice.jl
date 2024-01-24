# Tour of Dice.jl by example

################################################################################
# Basics
################################################################################

using Dice

# The fundamental datatype in Dice.jl is `Dist{Bool}`, which represents a
# distribution over booleans.

# `flip(p)` can be used create a random variable ~ Bernoulli(p)
x = flip(0.7)

# We can use `pr(dist)` on a distribution find the dictionary from possible
# values to probabilities.
pr(x)         # Dict(false => 0.3, true => 0.7)
pr(x | true)  # Dict(true => 1.0)

# We can use &, |, ^, ! to perform computation on DistBools.
pr(flip(0.5) & flip(0.1))  # true => 0.05

# We can find conditional probabilities using the `evidence` keyword arg
x = flip(0.5)
y = flip(0.1)
pr(x & y, evidence=x)  # Pr[X & Y given X] = 0.1

# The condition of `if` statements can be Dist{Bool} if written inside of the
# @dice_ite macro.
ite_example = @dice_ite begin
    if flip(0.9)
        true
    else
        !flip(0.5)
    end
end
pr(ite_example)  # true => 0.95


################################################################################
# Network Verification
################################################################################

# We now have enough to reconstruct the network verification example from the
# Dice paper.
# https://arxiv.org/pdf/2005.09089.pdf#page=6

function diamond(s1)
    @dice_ite begin
        route = flip(0.5)
        s2 = if route s1 else false end
        s3 = if route false else s1 end
        drop = flip(0.0001)
        # Functions in Julia implicitly return the value of the last expression
        s2 | (s3 & !drop)
    end
end

function network()
    net = true
    for _ in 1:10
        net = diamond(net)
    end
    net
end

net = network()
pr(net)  # true: 0.9995


################################################################################
# Ints
################################################################################

# DistUInts represent distributions over unsigned integers.
dist = pr(
    @dice_ite if flip(0.3)
        DistUInt32(7)
    else
        DistUInt32(3)
    end
)
dist  # Dict(7 => 0.3, 3 => 0.7)

# There is no magic here - DistInts are just a different way to interpret a
# joint distribution over multiple bools. We could also write the above as:
i = DistUInt{3}([flip(0.3), true, true]) # typed by width
                                         # most signif. bit first
pr(i)  # Dict(7 => 0.3, 3 => 0.7)

# A uniform distribution on integers in [start, stop]
x = uniform(DistUInt32, 1, 5)
y = uniform(DistUInt32, 1, 5)
pr(x)  # Dist(1 => 0.25, 2 => 0.25, 3 => 0.25, 4 => 0.25)

# To check for equality, use prob_equals
product_4 = prob_equals(x * y, DistUInt32(4))
pr(product_4)  # true => 0.1875 (3/16 cases: (1,4), (2,2), (4,1))

is_even(x::DistUInt32) = prob_equals(x % DistUInt32(2), DistUInt32(0))
pr(product_4, evidence=is_even(x))  # true => 0.25 (2/8 cases: (2,2), (4,1))
