using Dice
using BenchmarkTools

function fun()

    c = @dice begin 

        uniq_count = 8
        
        bits = Int(floor(log2(uniq_count)) + 4)
        partial = [[1, 2],
            [3, 2],
            [1, 4],
            [5, 8],
            [6, 5], 
            [3, 4]]

        ranks = [uniform(DistUIntOH{8}, 0, uniq_count) for i in 1:uniq_count]
        for i in 1:uniq_count
            for j in i+1:uniq_count
                observe(!prob_equals(ranks[i], ranks[j]))
            end
        end
        

        for comp in partial 
            observe(ranks[comp[1]] < ranks[comp[2]])
        end

        (ranks[1], ranks[2], ranks[3], ranks[4], ranks[5], ranks[6], ranks[7], ranks[8])
    end
    pr(c)
end 
x = @benchmark fun()

println((median(x).time)/10^9)