#==
"Dart target painting"
- We can paint a target using red, green, and blue
- Fixed numbers of red, green, and blue darts will randomly hit the target
- What proportion of paint colors maximizes the probability that at least one
  dart of each color lands in it own color?
==#

using Dice
import Base: all, any

all(itr) = reduce(&, itr)
any(itr) = reduce(|, itr)

DARTS_PER_COLOR = [1, 2, 10] # number of red, green, and blue darts
weights = [Var("r"), Var("g"), Var("b")]
var_vals = Valuation(weight => 1 for weight in weights)

all_colors_receive_own_dart = all(
  any(flip(weight / sum(weights)) for _ in 1:num_own_darts)
  for (num_own_darts, weight) in zip(DARTS_PER_COLOR, weights)
)

pr_mixed(var_vals)(all_colors_receive_own_dart) # 0.182
train!(var_vals, -LogPr(all_colors_receive_own_dart); epochs=1000, learning_rate=0.3)

# We've increased the chance of success!
pr_mixed(var_vals)(pr_all_colors_receive_own_dart) # 0.234

# Compute what ratio we actually need to paint the target:
[compute(var_vals, weight/sum(weights)) for weight in weights]
# 3-element Vector{Float64}:
#  0.46536681058883267
#  0.3623861813855715
#  0.17224700802559573
