
function gen_tree(s::Integer, lo::DistNat, hi::DistNat, track_return)
    track_return(
        @dice_ite if s == 0 || hi - lo < DistNat(2)
            DistE()
        else
            s′ = s - 1
            if flip(register_weight!("sz$(s)"))
                DistE()
            else
                k = unif(lo + DistNat(1), hi - DistNat(1))
                v = DistNat(0) # arbitrary
                l = gen_tree(s′, lo, k, track_return)
                r = gen_tree(s′, k, hi, track_return)
                DistT(l, k, v, r)
            end
        end
    )
end
