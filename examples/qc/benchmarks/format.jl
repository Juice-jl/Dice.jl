adnodes_of_interest = 

Dict("tysz2_gen_type_tbool" => 0.7207727519197863, "sz4_succ_var" => 0.4903234929252827, "tysz1_gen_type_tbool" => 0.5079797882722665, "sz0_zero_pr_var2" => 0.5, "sz2_succ_app" => 0.08352877939489839, "sz4_succ_abs" => 0.7066063985105583, "sz5_succ_var" => 0.5, "sz5_succ_abs" => 0.41721675684910503, "sz2_succ_var" => 0.5435857870063115, "sz1_succ_var" => 0.5460302801221573, "sz1_succ_app" => 0.1617094113296376, "sz1_succ_abs" => 0.6925946397620913, "sz3_succ_abs" => 0.7633559861209368, "sz3_succ_app" => 0.06947521647387661, "sz5_succ_app" => 0.5711467012997115, "sz4_succ_app" => 0.19063677116251065, "sz2_succ_abs" => 0.7423080357841665, "sz3_succ_var" => 0.5151953122355372)
@assert issetequal(keys(adnodes_of_interest), ["sz1_succ_abs", "tysz2_gen_type_tbool", "sz3_succ_abs", "sz4_succ_var", "sz3_succ_app", "sz5_succ_app", "tysz1_gen_type_tbool", "sz0_zero_pr_var2", "sz2_succ_app", "sz4_succ_abs", "sz5_succ_var", "sz4_succ_app", "sz2_succ_abs", "sz5_succ_abs", "sz3_succ_var", "sz2_succ_var", "sz1_succ_var", "sz1_succ_app"])

thousandths(n) = Integer(round(n, digits=3) * 1000)
w(s) = thousandths(adnodes_of_interest[s])

println("""genType
    let '(boolWeight, funWeight) :=
      get
      [
        (1, ($(w("tysz1_gen_type_tbool")), 1000-$(w("tysz1_gen_type_tbool"))));
        (2, ($(w("tysz2_gen_type_tbool")), 2000-$(w("tysz2_gen_type_tbool"))))
      ]

Fixpoint genExpr env tau (sz: nat) : G (option Expr) :=
  match sz with
  | 0 =>
    let '(var_weight, zero_weight) := ($(w("sz0_zero_pr_var2")), 1000-$(w("sz0_zero_pr_var2"))) in
      backtrack
        [(var_weight, oneOf_ (ret None) (map (fun x => returnGen (Some (Var x))) (genVar' env tau 0 [])))
        ;(zero_weight, genZero env tau)] 
  | S sz' =>
      let '(val_weight, app_weight, var_weight) :=
        get
          [
            (1, ($(w("sz1_succ_abs")), $(w("sz1_succ_app")), $(w("sz1_succ_var"))));
            (2, ($(w("sz2_succ_abs")), $(w("sz2_succ_app")), $(w("sz2_succ_var"))));
            (3, ($(w("sz3_succ_abs")), $(w("sz3_succ_app")), $(w("sz3_succ_var"))));
            (4, ($(w("sz4_succ_abs")), $(w("sz4_succ_app")), $(w("sz4_succ_var"))));
            (5, ($(w("sz5_succ_abs")), $(w("sz5_succ_app")), $(w("sz5_succ_var"))))
          ]
""")