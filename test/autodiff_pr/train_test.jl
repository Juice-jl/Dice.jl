using Test
using Dice

@testset "MLE" begin
    psp = Var("psp") # pre-sigmoid probability
    prob = sigmoid(psp)

    # Basic test
    x = flip(prob) & flip(prob) & !flip(prob)
    var_vals = Valuation(psp => 0)
    train!(var_vals, mle_loss([x]); epochs=1000, learning_rate=0.3)
    @test compute(var_vals, prob) ≈ 2/3

    # Try 1 - prob instead of negating the third flip
    x = flip(prob) & flip(prob) & flip(1 - prob)
    var_vals = Valuation(psp => 0)
    train!(var_vals, mle_loss([x]); epochs=1000, learning_rate=0.3)
    @test compute(var_vals, prob) ≈ 2/3

    # Approximate a dataset
    b = @dice_ite if flip(prob) true else flip(0.5) end
    dataset = [true, true, false]
    bools_to_maximize = [prob_equals(b, x) for x in dataset]
    var_vals = Valuation(psp => 0)
    train!(var_vals, mle_loss(bools_to_maximize); epochs=1000, learning_rate=0.3)
    @test compute(var_vals, prob) ≈ 1/3

    # Multiple params
    a_psp, b_psp, c_psp = Var("a_psp"), Var("b_psp"), Var("c_psp")
    a, b, c = sigmoid(a_psp), sigmoid(b_psp), sigmoid(c_psp)
    x = flip(a) & flip(b) & !flip(c)
    var_vals = Valuation(a_psp => 0, b_psp => 0, c_psp => 0)
    train!(var_vals, mle_loss([x]); epochs=1000, learning_rate=0.3)
    vals = compute(var_vals, [a, b, c])
    @test vals[a] > .99
    @test vals[b] > .99
    @test vals[c] < .01
end

@testset "MLE iterations deterministic" begin
    psp = Var("psp") # pre-sigmoid probability
    prob = sigmoid(psp)
    dataset = [true, true, false]

    # Train for 200 epochs
    var_vals = Valuation(psp => 0)
    b = @dice_ite if flip(prob) true else flip(prob) end
    train!(var_vals, mle_loss([prob_equals(b, x) for x in dataset]); epochs=200, learning_rate=0.003)
    p1 = pr_mixed(var_vals)(b)[true]

    # Train for 100 epochs, twice
    b = @dice_ite if flip(prob) true else flip(prob) end
    var_vals = Valuation(psp => 0)
    train!(var_vals, mle_loss([prob_equals(b, x) for x in dataset]); epochs=100, learning_rate=0.003)
    train!(var_vals, mle_loss([prob_equals(b, x) for x in dataset]); epochs=100, learning_rate=0.003)
    p2 = pr_mixed(var_vals)(b)[true]

    @test p1 ≈ p2
end

@testset "interleaving" begin
    x = Var("x")
    prob = sigmoid(x)
    prob2 = exp(LogPr(flip(prob) & flip(prob)))
    bool = flip(prob2) & flip(prob2) & !flip(prob2)
    loss = mle_loss([bool])
    var_vals = Valuation(x => 0)
    train!(var_vals, loss, epochs=2000, learning_rate=0.1)

    # loss is minimized if prob2 is 2/3
    # therefore, prob should be sqrt(2/3)
    @test compute(var_vals, prob) ≈ sqrt(2/3)
    @test compute_mixed(var_vals, loss) ≈ -log(2/3*2/3*1/3)
    pr_mixed(var_vals)(bool)
end
