using Test
using Dice

@testset "MLE" begin
    # Basic test
    prob = add_unit_interval_var!("?")
    x = flip(prob) & flip(prob) & !flip(prob)
    train_vars!([x]; epochs=1000, learning_rate=0.3)
    @test compute(prob) ≈ 2/3
    clear_vars!()

    # Try 1 - prob instead of negating the third flip
    prob = add_unit_interval_var!("?")
    x = flip(prob) & flip(prob) & flip(1 - prob)
    train_vars!([x]; epochs=1000, learning_rate=0.3)
    @test compute(prob) ≈ 2/3
    clear_vars!()

    # Approximate a dataset
    prob = add_unit_interval_var!("?")
    b = @dice_ite if flip(prob) true else flip(0.5) end
    dataset = [true, true, false]
    bools_to_maximize = [prob_equals(b, x) for x in dataset]
    train_vars!(bools_to_maximize; epochs=1000, learning_rate=0.3)
    @test compute(prob) ≈ 1/3
    clear_vars!()

    # Multiple params
    a = add_unit_interval_var!("a")
    b = add_unit_interval_var!("b")
    c = add_unit_interval_var!("c")
    x = flip(a) & flip(b) & !flip(c)
    train_vars!([x]; epochs=1000, learning_rate=0.3)
    @test compute(a) > .99
    @test compute(b) > .99
    @test compute(c) < .01
    clear_vars!()
end

@testset "MLE iterations deterministic" begin
    dataset = [true, true, false]
    
    # Train for 200 epochs
    prob = add_unit_interval_var!("?")
    b = @dice_ite if flip(prob) true else flip(prob) end
    train_vars!([prob_equals(b, x) for x in dataset]; epochs=200)
    p1 = pr(b)[true]
    clear_vars!()
    
    # Train for 100 epochs, twice
    prob = add_unit_interval_var!("?")
    b = @dice_ite if flip(prob) true else flip(prob) end
    train_vars!([prob_equals(b, x) for x in dataset]; epochs=100)
    train_vars!([prob_equals(b, x) for x in dataset]; epochs=100)
    p2 = pr(b)[true]
    clear_vars!()

    @test p1 ≈ p2
end
