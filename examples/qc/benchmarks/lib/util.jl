function backtrack_for(name, opts::Vector{DistI{Opt{T}}})::DistI{Opt{T}} where T
    first_some(T, shuffle_for(name, opts))
end

function shuffle_for(name, xs)
    # Hand-build shuffle for lengths 2 and 3
    @dice_ite if length(xs) == 2
        pr_var2 = register_weight!("$(name)_pr_var2")
        if flip(pr_var2)
            [xs[1], xs[2]]
        else
            [xs[2], xs[1]]
        end
    elseif length(xs) == 3
        var = register_weight!("$(name)_var")
        app = register_weight!("$(name)_app")
        val = register_weight!("$(name)_abs")
        if flip(var / (var + app + val))
            # var is first
            if flip(app / (app + val))
                [xs[1], xs[2], xs[3]]  # var app val
            else
                [xs[1], xs[3], xs[2]]  # var val app
            end
        elseif flip(app / (app + val))
            # app is first
            if flip(var / (var + val))
                [xs[2], xs[1], xs[3]]  # app var val
            else
                [xs[2], xs[3], xs[1]]  # app val var
            end
        else
            # val is first
            if flip(var / (var + app))
                [xs[3], xs[1], xs[2]]  # val var app
            else
                [xs[3], xs[2], xs[1]]  # val app var
            end
        end
    else
        error("todo: generic shuffle")
    end
end

function frequency(xs)
    sample_from(xs)
end

function backtrack(xs)
    isempty(xs) && return DistNone()
    x = sample_from(xs)
    remove!(xs, x)
    backtrack(xs)
end

function first_some(::Type{T}, xs) where T
    isempty(xs) && return DistNone(T)
    x, rest = xs[1], @view xs[2:end]
    @dice_ite if matches(x, "Some")
        x
    else
        first_some(T, rest)
    end
end

# Manually curry so we can have type be first arg and use "do"
function map(::Type{RetT}) where RetT
    function inner(f, l::DistI{List{T}}) where T
        match(l, [
            "Nil" => () -> DistNil(RetT),
            "Cons" => (x, xs) -> DistCons(f(x), map(RetT)(f, xs))
        ])
    end
end

function frequency_for(name, xs)
    # Hand-build shuffle for lengths 2 and 3
    @dice_ite if length(xs) == 2
        if flip_for("$(name)_var_")
            xs[1]
        else
            xs[2]
        end
    elseif length(xs) == 3
        if flip_for("$(name)_var", 1/3)
            xs[1]
        elseif flip_for("$(name)_app")
            xs[2]
        else
            xs[3]
        end
    else
        error("todo: generic frequency_for")
    end
end

function opt_stlc_str(ast)
    name, children = ast
    if name == "None"
        "None"
    elseif name == "Some"
        ast′, = children
        stlc_str(ast′)
    else
        error("Bad node $(name)")
    end
end

function save_metric_dist(filename, dist; io)
    open(filename, "w") do file
        println(file, "val\tprobability")
        for i in key_range(dist)
            println(file, "$(i)\t$(dist[i])")
        end
    end
    println(io, "Saved metric dist to $(filename).")
end

key_range(d) = minimum(keys(d)):maximum(keys(d))

function preview_distribution(e; full_dist)
    if full_dist
        println("Getting distribution of all exprs")
        @time dist = pr(e)
        for (k, pr) in dist
            println("pr: $(pr)")
            println(opt_stlc_str(k))
            println()
        end
    else
        println("A few sampled exprs:")
        for _ in 1:20
            expr = sample(e)
            println(opt_stlc_str(expr))
        end
    end
end

function save_samples(filename, e; n_samples=200, io=stdout)
    open(filename, "w") do file
        for _ in 1:n_samples
            expr = sample(e)
            println(file, opt_stlc_str(expr))
            typecheck_opt(expr)
        end
    end
    println(io, "Saved samples to $(filename).")
end

function println_flush(io, args...)
    println(io, args...)
    flush(io)
end
