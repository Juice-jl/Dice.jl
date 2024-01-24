using Dice
using Test

@testset "test autodiff" begin
    x = Var("x")

    e = x^2
    _, derivs = differentiate(Valuation(x => 5), Derivs(e => π))
    @test derivs[x] ≈ 10 * π

    e = 1/x
    _, derivs = differentiate(Valuation(x => 5), Derivs(e => π))
    @test derivs[x] ≈ -1/25 * π
end

@testset "test GD" begin
    x, y = Var("x"), Var("y")

    var_vals = Valuation(x => 0)
    e = 7 - (x - 5)^2
    train!(var_vals, -e, epochs=100, learning_rate=0.1)
    vals = compute(var_vals, [e])
    @test vals[e] ≈ 7
    @test vals[x] ≈ 5

    var_vals = Valuation(x => 7, y => 1)
    e = -(x*y-5)^2
    train!(var_vals, -e, epochs=100, learning_rate=0.01)
    vals = compute(var_vals, [e])
    @test abs(vals[e]) < 0.001
    @test vals[x] * vals[y] ≈ 5
    
    psp = Var("psp") # pre-sigmoid probability
    var_vals = Valuation(psp => 0)
    p = sigmoid(psp)
    # maximize logpr of flip(p) & flip(p) & !flip(p)
    e = log(p * p * (1 - p))
    train!(var_vals, -e, epochs=500, learning_rate=0.1)
    @test compute(var_vals, p) ≈ 2/3
end

@testset "matrices" begin
    # Helper functions for matrices used as vectors
    x(v) = v[1,1]
    y(v) = v[2,1]
    distance(u, v) = (x(u) - x(v))^2 + (y(u) - y(v))^2
    to_matrix(v::Vector) = reshape(v, :, 1)

    # Rotate [1, 2] by what angle to get closest to [-3, -3]?
    θ = Var("θ")
    var_vals = Valuation(θ => 0)
    rotation_matrix = ADMatrix([[cos(θ) -sin(θ)]; [sin(θ) cos(θ)]])
    rotated_vec = rotation_matrix * to_matrix([1, 2])
    target_vec = to_matrix([-3, -3])
    train!(var_vals, distance(rotated_vec, target_vec), epochs=2000, learning_rate=0.003)
    @test var_vals[θ] ≈ 5/8 * 2π - atan(2)

    # Variables can also be matrices!
    # Transform by [1, 2] by what matrix to get closest to [-3, -3]?
    A = Var("A")
    var_vals = Valuation(A => [[1 0]; [0 1]])
    v = to_matrix([1, 2])
    v′ = A * v
    target_vec = to_matrix([-3, -3])
    train!(var_vals, distance(v′, target_vec), epochs=2000, learning_rate=0.003)
    @test var_vals[A] * v ≈ [-3, -3]
end
