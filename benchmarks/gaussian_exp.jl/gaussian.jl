using Revise
using Dice
using Dice: num_flips, num_nodes, to_dice_ir
using Distributions
using StatsBase
using Distances
using Plots; gr()

function KL_div(a, d::ContinuousUnivariateDistribution, upper, lower, interval)
    d = Truncated(d, lower, upper)
    len = Int((upper - lower)/interval)
    lower2 = a[1][1]
    upper2 = a[length(a)][1]
    lower_index = 1
    upper_index = length(a)
    for i = 1:length(a)
        if a[i][2] == 0.0
            lower2 = a[i+1][1]
            lower_index = i+1
        else
            break
        end
    end
    for i = length(a):-1:1
        if a[i][2] == 0.0
            upper2 = a[i-1][1]
            upper_index = i-1
        else
            break
        end
    end
    len = Int((upper2 - lower2)/interval)
    p = Vector{Float64}(undef, upper_index - lower_index + 1)
    for i=1:upper_index - lower_index + 1
        @show lower2 + i*interval, lower2 + (i-1)*interval
        p[i] = cdf(d, lower2 + i*interval) - cdf(d, lower2 + (i-1)*interval)
    end
    # @show upper_index, lower_index
    # @show lower2 + (upper_index - lower_index)*interval, upper2
    @assert lower2 + (upper_index - lower_index)*interval == upper2
    # @show upper2, lower2
    @show a[lower_index:upper_index]
    kldivergence(p, map(a->a[2], a[lower_index: upper_index]))
end

function gaussian(p::Int, binbits::Int, flag::Int)
    code = @dice begin
        t = DistSigned{binbits+5, binbits}
        a = continuous(dicecontext(), p, t, Normal(0, 1), flag)
        # prob_equals(a, t(dicecontext(), -2.0))
        a
    end
    code
end

for i = 0:4
    a = gaussian(2^i, 0, 1)
end

flag = 0
x = Vector(undef, 128)
y = Vector(undef, 6)
for i = 0:4
    a = gaussian(2^i, 0, flag)
    b = infer(a, :bdd)
    # @show b
    # @show KL_div(b, Normal(0, 1), 16, -16, 1)
    x = map(a -> a[1], b)
    y[i+1] = map(a -> a[2], b)
end
d = Normal(0, 1)
z = Vector(undef, 32)
for i=1:32
    z[i] = cdf(d, i - 16) - cdf(d, i - 17)
end
y[6] = z
plot(x, y, labels = [1 2 4 8 16 "gt"])

flag = [0 1 3]
x = [0; 1; 2; 3; 4; 5; 6]
y = Vector(undef, 3)
for fl in 1:length(flag)
    z = Vector(undef, 7)
    for i = 0:6
        a = gaussian(2^i, 2, flag[fl])
        b = infer(a, :bdd)
        # z[i+1] = KL_div(b, Normal(0, 1), 16, -16, 0.25)
        z[i+1] = abs(expectation(a, :bdd) + 0.125) + 0.125
        # z[i+1] = abs(variance(a, :bdd) - 1)
    end
    y[fl] = z
end
plot(x, y, labels = ["cdf" "pdf" "cdf of halves"], yaxis=:log, legend=:topright)

function gaussian_uniform(p::Int, binbits::Int, b::Bool, b2::Bool, flag::Int)
    code = @dice begin
        t = DistSigned{binbits+5, binbits}
        a = add_bits(continuous(dicecontext(), p, t, Normal(0, 1), b, b2), 1, 0)
        mu = add_bits((uniform(dicecontext(), binbits+4, t) + t(dicecontext(), -8.0))[1], 1, 0)
        d = (a + mu)[1]
        prob_equals(d, DistSigned{binbits + 6, binbits}(dicecontext(), -12.0))
    end
    code
end

a = gaussian_uniform(2, 0, false, false, 1)
b = infer(a, :bdd)

x = Vector(undef, 32)
y = Vector(undef, 5)
for i = 0:4
    a = gaussian_uniform(2^i, 0, false, false, 1)
    b = infer(a, :bdd)
    @show b
    # x = map(a -> a[1], b)
    # y[i+1] = map(a -> a[2], b)
end
# d = Normal(0, 1)
# z = Vector(undef, 32)
# for i=1:32
#     z[i] = cdf(d, i - 16) - cdf(d, i - 17)
# end
# y[6] = z
plot(x, y)

function gaussian_two_uniform(p::Int, binbits::Int, b::Bool, b2::Bool, flag::Int)
    code = @dice begin
        t = DistSigned{binbits+5, binbits}
        a1 = add_bits(continuous(dicecontext(), p, t, Normal(0, 1), b, b2), 2, 0)
        a2 = add_bits(continuous(dicecontext(), p, t, Normal(0, 1), b, b2), 2, 0)
        mu1 = add_bits((uniform(dicecontext(), binbits+4, t) + t(dicecontext(), -8.0))[1], 1, 0)
        mu2 = add_bits((uniform(dicecontext(), binbits+4, t) + t(dicecontext(), -8.0))[1], 1, 0)
        d1 = (add_bits((mu1 + mu2)[1], 1, 0) + a1)[1]
        d2 = (add_bits((mu1 + mu2)[1], 1, 0) + a2)[1]
        # DistSigned{binbits + 6, binbits}(d1.number.bits[1:binbits + 6])
        prob_equals(d1, DistSigned{binbits + 7, binbits}(dicecontext(), -15.0)) 
        # & 
        # prob_equals(d2, DistSigned{binbits + 7, binbits}(dicecontext(), 0.33806206))
    end
    code
end

# x = Vector(undef, 32)
# y = Vector(undef, 5)
for i = 0:4
    a = gaussian_two_uniform(2^i, 0, false, false, 1)
    b = infer(a, :bdd)
    @show b
    # x = map(a -> a[1], b)
    # y[i+1] = map(a -> a[2], b)
end
# d = Normal(0, 1)
# z = Vector(undef, 32)
# for i=1:32
#     z[i] = cdf(d, i - 16) - cdf(d, i - 17)
# end
# y[6] = z
plot(x, y)

function uniform_gaussian_bits(p::Int, binbits::Int, b::Bool, b2::Bool, flag::Int)
    code = @dice begin
        t = DistFixParam{binbits+3, binbits}
        mu = add_bits(uniform(dicecontext(), 1, t), 1)
        d = if flag == 1 Normal(3, 1) elseif flag == 2 Normal(0, 3) else Normal(8, 3) end
        a = add_bits(continuous(dicecontext(), p, t, d, b, b2), 1)
        # mu = uniform(dicecontext(), 1, t)
        (a + mu)[1]
    end
    code
end

for i = 0:10
    for j = 0:i+2
        f = uniform_gaussian_bits(2^j, i, false, false, 1)
        a = compile(f)
        @show i, j, num_flips(a), num_nodes(a)
    end
end

function gaussian_uniform_bits(p::Int, binbits::Int, b::Bool, b2::Bool, flag::Int)
    code = @dice begin
        t = DistFixParam{binbits+3, binbits}
        # mu = uniform(dicecontext(), 1, t)
        d = if flag == 1 Normal(3, 1) elseif flag == 2 Normal(0, 3) else Normal(8, 3) end
        a = add_bits(continuous(dicecontext(), p, t, d, b, b2), 1)
        mu = add_bits(uniform(dicecontext(), 1, t), 1)
        (a + mu)[1]
    end
    code
end

for i = 0:10
    for j = 0:i+2
        f = gaussian_uniform_bits(2^j, i, false, false, 1)
        a = compile(f)
        @show i, j, num_flips(a), num_nodes(a)
    end
end

function gaussian_uniform_eq(p::Int, binbits::Int, b::Bool, b2::Bool, flag::Int)
    code = @dice begin
        t = DistFixParam{binbits+3, binbits}
        d = if flag == 1 Normal(3, 1) elseif flag == 2 Normal(0, 3) else Normal(8, 3) end
        a = add_bits(continuous(dicecontext(), p, t, d, b, b2), 1)
        mu = add_bits(uniform(dicecontext(), binbits+3, t), 1)
        prob_equals((a + mu)[1], DistFixParam{binbits + 4, binbits}(dicecontext(), 2.0))
    end
    code
end

for i = 0:10
    for j = 0:i+2
        f = gaussian_uniform_eq(2^j, i, false, false, 1)
        a = compile(f)
        @show i, j, num_flips(a), num_nodes(a), infer(a)
    end
end

bits = 0
pieces = 1
f = gaussian_uniform_eq(pieces, bits, false, false, 1)
a = compile(f)
num_nodes(a)
num_flips(a)
infer(a)
dump_dot(a, "rough.dot")

bits = 0
pieces = 2
f = gaussian_uniform_eq(pieces, bits, false, false, 1)
a = compile(f)
num_nodes(a)
num_flips(a)
infer(a)
dump_dot(a, "rough2.dot")

bits = 0
pieces = 4
f = gaussian_uniform_eq(pieces, bits, false, false, 1)
a = compile(f)
num_nodes(a)
num_flips(a)
infer(a)
dump_dot(a, "rough3.dot")

 function uniform_gaussian_eq(p::Int, binbits::Int, b::Bool, b2::Bool, flag::Int)
    code = @dice begin
        t = DistFixParam{binbits+3, binbits}
        d = if flag == 1 Normal(3, 1) elseif flag == 2 Normal(0, 3) else Normal(8, 3) end
        mu = uniform(dicecontext(), binbits+3, t)
        a = continuous(dicecontext(), p, t, d, b, b2)
        prob_equals((a + mu)[1], t(dicecontext(), 0.0))
    end
    code
end

for i = 0:10
    for j = 0:i+2
        f = uniform_gaussian_eq(2^j, i, false, false, 1)
        a = compile(f)
        @show i, j, num_flips(a), num_nodes(a)
    end
end

x = Vector(undef, 13)
y = Vector(undef, 13)
for j = 0:12
    f = gaussian_uniform_eq(2^j, 10, false, false, 1)
    a = compile(f)
    x[j+1] = j
    y[j+1] = num_nodes(a)
end
plot(x, y)

bits = 1
pieces = 1
f = gaussian_uniform_eq(pieces, bits, false, false, 1)
a = compile(f)
num_nodes(a)
num_flips(a)
infer(a)
dump_dot(a, "rough.dot")

bits = 1
pieces = 2
f = gaussian_uniform_eq(pieces, bits, false, false, 1)
a = compile(f)
num_nodes(a)
num_flips(a)
infer(a)
dump_dot(a, "rough2.dot")

bits = 1
pieces = 4
f = gaussian_uniform_eq(pieces, bits, false, false, 1)
a = compile(f)
num_nodes(a)
num_flips(a)
infer(a)
dump_dot(a, "rough3.dot")

bits = 1
pieces = 8
f = uniform_gaussian_eq(pieces, bits, false, false, 1)
a = compile(f)
num_nodes(a)
num_flips(a)
infer(a)
dump_dot(a, "rough3.dot")


