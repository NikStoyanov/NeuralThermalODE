using Test
using NeuralThermalODE

# Test data file.
# The CSV format is:
# | Time | Temperature |
# ----------------------
# | INT  | FLOAT       |
# | ...  | ...         |
test_data = read_data("test_data/test_data.csv")

# Domain.
const N = 10
const tmat = 0.00635 # Steel thickness.
const dx = tmat / N  # Element size.

# Boundary conditions from the test data.
const sinkT = test_data[end, 2] # Sink temperature on exposed side.
const ambT = test_data[1, 2]    # Sink temperature on unexposed side.
u20 = ambT * ones(N)            # Initial temperature.

const tspan = (Float64(test_data[1, 1]), Float64(test_data[2, 1]))

# For unexposed HTC formula for ambient see:
# /home/nik/Dropbox/PhD/Academic/General_Data/HTC_effect/Unexposed
# This might need to be investigated to see if 4.0 is good enough.
const pr = BuildThermalODE(dx, 450.0, 7850.0, 42.0, sinkT, ambT, 4.0)

# Solve ODE.
p = read_checkpoint("test_data/initial_guess.csv")

# Pass additional arguments through closure.
prob = ODEProblem((du, u20, p, t) -> heat_transfer(pr, du, u20, p, t), u20, tspan, p, saveat = 1.0)

data = Iterators.repeated((), 5)
opt = ADAM(20.0)

target_loss = 10.0

# Track loss function.
tot_loss = []

training_cnt = -1
early_exit = false

algo = Tsit5()

cb = function ()
    global training_cnt += 1

    # Save loss function.
    loss = loss_rd(Vector(test_data[2]), p, prob, algo)
    append!(tot_loss, loss)

    if loss <= target_loss
        early_exit = true
        try
            Flux.stop()
        catch
        end
    end

    sol = solve(remake(prob, p = Flux.data(p)), algo, saveat = 1.0)
end

cb()
Flux.train!(() -> loss_rd(Vector(test_data[2]), p, prob, algo), [p], data, opt, cb = cb)

res = Flux.data(p)[1:end-1]
@test isapprox(res[1], 706.0, atol = 1.0)
