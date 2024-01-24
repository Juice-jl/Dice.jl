module Dice

##################################
# Control flow macro
##################################

using MacroTools: prewalk, postwalk
using PrecompileTools

export @dice_ite

"Syntactic macro to make if-then-else supported by dice"
macro dice_ite(code)
    postwalk(esc(code)) do x
        if x isa Expr && (x.head == :if || x.head == :elseif)
            @assert length(x.args) == 3 "@dice_ite macro only supports purely functional if-then-else"
            ite_guard = gensym(:ite)
            return :(begin $ite_guard = $(x.args[1])
                    if (!Dice.indynamo() && $(ite_guard) isa Dist{Bool})
                        ifelse($(ite_guard), $(x.args[2:3]...))
                    else
                        (if $(ite_guard)
                            $(x.args[2])
                        else
                            $(x.args[3])
                        end)
                    end
                end)
        end
        return x
    end
end

include("autodiff/adnode.jl")
include("autodiff/core.jl")
include("dist/dist.jl")
include("inference/inference.jl")
include("autodiff_pr/train.jl")
include("autodiff_pr/losses.jl")
include("analysis/analysis.jl")
include("dsl.jl")
include("plot.jl")
include("util.jl")

# add precompile statements here - keep it lightweight for main features
@compile_workload begin
    code = @dice begin 
        if flip(0.5) true else flip(0.1) end 
    end
    pr(code)
end

end # module