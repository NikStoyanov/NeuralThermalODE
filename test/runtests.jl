using Test
using NeuralThermalODE

# Test data file.
test_data = read_data("test_data/test_data.csv")

# Domain.
const N = 10
const t = 0.00635
const dx = t / N

# Boundary conditions.
const sinkT = test_data[end, 2]
const ambT = test_data[1, 2]
u20 = ambT * ones(N)
const tspan = (Float64(test_data[1, 1]), Float64(test_data[2, 1]))

const pr = BuildThermalODE(450.0, 7850.0, 42.0, sinkT, ambT, 4.0)

# Solve ODE.
p = read_checkpoint("test_data/initial_guess.csv")

du = zeros(N)

# Pass additional arguments through closure.
prob = ODEProblem((u20, p, t) -> heat_transfer(pr, du, u20, p, tspan), u20, tspan, p, saveat = 1.0)

data = Iterators.repeated((), 10000)
opt = ADAM(20.0)

# Readings with plus/minus 2.0C each
target_loss = 2 * 2.2

# Track loss function.
tot_loss = []

training_cnt = -1
early_exit = false

cb = function ()
    global training_cnt += 1

    # Save loss function.
    loss = loss_rd()
    append!(tot_loss, loss)

    if loss <= target_loss
        early_exit = true
        Flux.stop()
    end

    sol = solve(remake(prob, p = Flux.data(p)), Tsit5(), saveat = 1.0)
end

cb()
Flux.train!(loss_rd, [p], data, opt, cb = cb)

@test early_exit == true

#df = DataFrame(x1 = Float64(test_data[1, 1]):Float64(test_data[end, 1]),
#            x2 = Flux.data(p)[1:end-1])
#write_checkpoint("last_checkpoint_htc_$training_cnt.csv", df)
