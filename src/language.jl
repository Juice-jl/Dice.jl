export DiceProgram

struct DiceProgram
    expr
end

struct Flip
    prob::Float64    
end

struct Categorical
    probs::Vector{Float64}    
end

struct Identifier
    symbol::String    
end

struct EqualsOp
    e1
    e2    
end

struct Ite
    cond_expr
    then_expr
    else_expr
end

struct LetExpr
    identifier::Identifier
    e1
    e2 
end
