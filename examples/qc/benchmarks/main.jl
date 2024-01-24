# Train an STLC generator to have some property (term size or num apps) match a
# specific distribution (linear or uniform).
#
# Saves the distributions and sampled terms before and after training.

using Dice
include("lib/lib.jl")
include("benchmarks.jl")

############################
# Config
############################

generation_params = STLCGenerationParams(
    param_vars_by_size=true,
    size=5,
    ty_size=2,
)
loss_params = STLC4321AppsLoss()

# generation_params = BSTGenerationParams(size=3)
# loss_params = ApproxBSTConstructorEntropy()

EPOCHS = 500
LEARNING_RATE = if loss_params isa ApproxSTLCConstructorEntropy 0.03 else 0.01 end

TAG = "v04_infra"

LOG_TO_FILE = true

############################

path = joinpath(
    vcat(
        [TAG],
        to_subpath(generation_params),
        to_subpath(loss_params),
        ["epochs=$(EPOCHS),learning_rate=$(LEARNING_RATE)"],
    )
)
OUT_DIR = "examples/qc/benchmarks/output/$(path)"

###########################

LOG_PATH = joinpath(OUT_DIR, "log.log")
LEARNING_CURVE_PATH = joinpath(OUT_DIR, "learning_curve.csv")

mkpath(OUT_DIR)
io = if LOG_TO_FILE open(LOG_PATH, "w") else stdout end

using Dates
t = now()
for io′ in Set([io, stdout])
    println(io′, t)
    println(io′, "== Config ==")
    println(io′, "TAG: $(TAG)")
    println(io′, generation_params)
    println(io′, loss_params)
    println(io′, "EPOCHS: $(EPOCHS)")
    println(io′, "LEARNING_RATE: $(LEARNING_RATE)")
    println(io′, "DistNat: $(DistNat)")
    println(io′)
end
if LOG_TO_FILE
    println("Logging to $(LOG_PATH)")
    println()
end

var_vals = Valuation()
adnodes_of_interest = Dict{String, ADNode}()
function register_weight!(s)
    var = Var("$(s)_before_sigmoid")
    if !haskey(var_vals, var) || var_vals[var] == 0
        var_vals[var] = 0
    else
        println(io, "WARNING: not registering fresh weight for $(s)")
    end
    weight = sigmoid(var)
    adnodes_of_interest[s] = weight
    weight
end

println_flush(io, "Building generation computation graph...")
time_build_generation = @elapsed generation = generate(generation_params)
println(io, "  $(time_build_generation) seconds")
println(io)

println_flush(io, "Building generation loss computation graph...")
time_build_loss = @elapsed loss, extra = build_loss(loss_params, generation)
println(io, "  $(time_build_loss) seconds")
println(io)


############################
# Before
############################

println(io, "Initial var_vals:")
show(io, var_vals)
println(io)
println(io)

println(io, "Initial adnodes_of_interest:")
vals = compute(var_vals, values(adnodes_of_interest))
show(io, Dict(s => vals[adnode] for (s, adnode) in adnodes_of_interest))
println(io)
println(io)

if loss_params isa MLELossParams{STLC}
    println_flush(io, "Inferring initial distribution...")
    time_infer_init = @elapsed metric_dist = pr_mixed(var_vals)(extra)
    println(io, "  $(time_infer_init) seconds")
    save_metric_dist(joinpath(OUT_DIR, "dist_before.csv"), metric_dist; io)
    println(io)
end

if generation isa STLCGeneration
    println_flush(io, "Saving samples...")
    time_sample_init = @elapsed with_concrete_ad_flips(var_vals, generation.e) do
        save_samples(joinpath(OUT_DIR, "terms_before.txt"), generation.e; io=io)
    end
    println(io, "  $(time_sample_init) seconds")
    println(io)
end

initial_loss = compute_mixed(var_vals, loss)
println(io, "Initial loss: $(initial_loss)")
println(io)

############################
# Train
############################

println_flush(io, "Training...")
time_train = @elapsed learning_curve = train!(var_vals, loss; epochs=EPOCHS, learning_rate=LEARNING_RATE)
println(io, "  $(time_train) seconds")
println(io)

open(LEARNING_CURVE_PATH, "w") do file
    for (epoch, logpr) in zip(0:EPOCHS, learning_curve)
        println(file, "$(epoch)\t$(logpr)")
    end
end

############################
# After
############################

final_loss = compute_mixed(var_vals, loss)
println(io, "Final loss: $(final_loss)")
if loss_params isa MLELossParams
    approx_improvement = round(exp(initial_loss - final_loss), digits=2)
    println(io, "Drawing the target dataset is $(approx_improvement)x more likely")
end
println(io)

println(io, "Learned var_vals:")
show(io, var_vals)
println(io)
println(io)

println(io, "Learned adnodes_of_interest:")
vals = compute(var_vals, values(adnodes_of_interest))
show(io, Dict(s => vals[adnode] for (s, adnode) in adnodes_of_interest))
println(io)
println(io)

if generation isa STLCGeneration
    println(io, "Inferring trained num apps distribution...")
    time_infer_final = @elapsed metric_dist_after = pr_mixed(var_vals)(num_apps(generation.e))
    save_metric_dist(joinpath(OUT_DIR, "dist_trained.csv"), metric_dist_after; io)
    println(io, "  $(time_infer_final) seconds")
    println(io)

    println(io, "Saving samples...")
    time_sample_final = @elapsed with_concrete_ad_flips(var_vals, generation.e) do
        save_samples(joinpath(OUT_DIR, "terms_trained.txt"), generation.e; io)
    end
    println(io, "  $(time_sample_final) seconds")
    println(io)
end
