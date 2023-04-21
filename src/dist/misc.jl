##################################
# Nothing
##################################

tobits(::Nothing) = []
frombits(::Nothing, _) = nothing

##################################
# Tuple
##################################

tobits(x::Tuple) = 
    mapreduce(tobits, vcat, x)

frombits(x::Tuple, world) = 
    map(v -> frombits(v, world), x)

tobits(x::Matrix) = 
    mapreduce(tobits, vcat, x)

frombits(x::Matrix, world) = 
    map(v -> frombits(v, world), x)

Base.ifelse(cond::Dist{Bool}, then::Tuple, elze::Tuple) =
    Tuple(ifelse(cond, x, y) for (x, y) in zip(then,elze))

tobits(x::Vector) = 
    mapreduce(tobits, vcat, x)

frombits(x::Vector, world) = 
    map(v -> frombits(v, world), x)

Base.ifelse(cond::Dist{Bool}, then::Vector, elze::Vector) =
    Vector(ifelse(cond, x, y) for (x, y) in zip(then,elze))


