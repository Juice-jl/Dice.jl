# Define DistUTLC
import Dice: param_lists

struct DistUTLC <: InductiveType end

function param_lists(::Type{DistUTLC})::Vector{Pair{String,Vector{Type}}}
    [
        "Var" => [DistString],
        "App" => [DistI{DistUTLC}, DistI{DistUTLC}],
        "Abs" => [DistString, DistI{DistUTLC}],
    ]
end

DistVar(s)      = construct(DistUTLC, "Var", [s,])
DistApp(e1, e2) = construct(DistUTLC, "App", [e1, e2])
DistAbs(s, e)   = construct(DistUTLC, "Abs", [s, e])

function ast_depth(l::DistI{DistUTLC})
    match(l, [
        "Var"    => (s)      -> DistUInt32(0),
        "App"    => (e1, e2) -> begin
            d1, d2 = ast_depth(e1), ast_depth(e2)
            @dice_ite if d1 > d2
                DistUInt32(1) + d1
            else
                DistUInt32(1) + d2
            end
        end,
        "Abs"    => (s, e)   -> DistUInt32(1) + ast_depth(e),
    ])
end

# https://stackoverflow.com/questions/59338968/printing-lambda-expressions-in-haskell

parens(b, s) = if b "($(s))" else s end

@enum Ctx free=0 fun=1 arg=2

function utlc_str(ast, p=free)
    name, children = ast
    if name == "Var"
        s, = children
        s
    elseif name == "Abs"
        s, e = children
        parens(p > free, "λ$(s).$(utlc_str(e, free))")
    elseif name == "App"
        e1, e2 = children
        parens(
            p > fun,
            "$(utlc_str(e1, fun)) $(utlc_str(e2, arg))"
        )
    else
        error("Bad node $(name)")
    end
end
