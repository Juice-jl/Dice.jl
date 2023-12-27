using Test
using Dice
using Plots

function dict_approx(d1::AbstractDict, d2::AbstractDict)
    keys(d1) == keys(d2) && all(
        d1[k] ≈ d2[k]
        for k in keys(d1)
    )
end

@testset "learned_cheap_dists" begin
    function dist_binomial(width, p)
        n = 2^width - 1
        Dict(
            k => binomial(n, k) * p^k * (1 - p)^(n - k)
            for k in 0:n
        )
    end


    width = 3
    target_dist = dist_binomial(width, 0.1)

    vars = [Var("var$(i)_psp") for i in 0:width-1]
    int = DistUInt{width}([flip(sigmoid(x)) for x in vars])

    # Weighted training
    wt_var_to_vals = Valuation(x => 0 for x in vars)
    history = train_pr!(
        wt_var_to_vals,
        mle_loss([
            BoolToMax(
                prob_equals(int, DistUInt{width}(i)),
                weight=target_dist[i]
            )
            for i in 0:2^width-1
        ]),
        epochs=300,
        learning_rate=0.1
    )
    wt_dist = pr(int, flip_pr_resolver=valuation_to_flip_pr_resolver(wt_var_to_vals))
    @test dict_approx(
        wt_var_to_vals,
        Dict(
            Var("var0_psp") => -3.2837129282982764,
            Var("var1_psp") => -1.734139957995633,
            Var("var2_psp") => -0.4254578347528184)
    )
    @test dict_approx(
        wt_dist,
        Dict(
            0 => 0.4954604613881238,
            1 => 0.3237688128294564,
            2 => 0.08747452404585239,
            3 => 0.057162024036790764,
            4 => 0.018574220540662965,
            5 => 0.012137705835969861,
            6 => 0.0032793153600291173,
            7 => 0.002142935963114733
        )
    )

    # KL divergence minimization
    kl_var_to_vals = Valuation(x => 0 for x in vars)
    train_pr!(
        kl_var_to_vals,
        kl_divergence(
            target_dist,
            int,
            Set([i => DistUInt{width}(i) for i in 0:2^width-1])
        ),
        epochs=300,
        learning_rate=0.1
    )
    kl_dist = pr(int, flip_pr_resolver=valuation_to_flip_pr_resolver(kl_var_to_vals))
    @test dict_approx(
        wt_var_to_vals,
        Dict(
            Var("var0_psp") => -3.2837129282982764,
            Var("var1_psp") => -1.734139957995633,
            Var("var2_psp") => -0.4254578347528184)
    )
    @test dict_approx(
        wt_dist,
        Dict(
            0 => 0.4954604613881238,
            1 => 0.3237688128294564,
            2 => 0.08747452404585239,
            3 => 0.057162024036790764,
            4 => 0.018574220540662965,
            5 => 0.012137705835969861,
            6 => 0.0032793153600291173,
            7 => 0.002142935963114733
        )
    )
end
