# Define Tree
import Dice: param_lists

struct Tree <: InductiveType end

function param_lists(::Type{Tree})::Vector{Pair{String,Vector{Type}}}
    [
        "E" => [],
        "T" => [DistI{Tree}, DistNat, DistNat, DistI{Tree}],
    ]
end

DistE() = construct(Tree, "E",   [])
DistT(l::DistI{Tree}, k::DistNat, v::DistNat, r::DistI{Tree}) =
    construct(Tree, "T", [l, k, v, r])

bst_ctor_to_id = Dict(
    "E" => DistInt32(0),
    "T" => DistInt32(1),
)

function ctor_to_id(ctor::DistI{Tree})
    match(ctor, [
        "E" => () -> bst_ctor_to_id["E"]
        "T" => (_, _, _, _) -> bst_ctor_to_id["T"]
    ])
end
