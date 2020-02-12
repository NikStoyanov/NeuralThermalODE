using NeuralThermalODE

# Test data file.
test_data = read_data("Fig_67.csv_Result.csv")

# Domain.
const N = 10
const tmat = 0.00635
const dx = tmat / N

# Boundary conditions.
const sinkT = test_data[end, 2]
const ambT = test_data[1, 2]
u20 = ambT * ones(N)
const tspan = (Float64(test_data[1, 1]), Float64(test_data[end, 1]))

const pr = BuildThermalODE(450.0, 7850.0, 42.0, sinkT, ambT, 4.0)

# Solve ODE.
p = read_checkpoint("initial_guess.csv")

du = zeros(N)

# Pass additional arguments through closure.
prob = ODEProblem((du, u20, p, t) -> heat_transfer(pr, du, u20, p, t), u20, tspan, p, saveat = 1.0)

data = Iterators.repeated((), 10000)
opt = ADAM(20.0)

# 180 readings with plus/minus 2.0C each
target_loss = 180.0 * 2.2

# Track loss function.
tot_loss = []

training_cnt = -1
cb = function ()
    early_exit = false
    save_curr = false

    global training_cnt += 1

    if training_cnt % 10 == 0
        save_curr = true
    end

    # Save loss function.
    loss = loss_rd()
    append!(tot_loss, loss)

    print("Loss $loss ")
    println("Iteration $training_cnt")

    if loss <= target_loss
        save_curr = true
        early_exit = true
    end

    sol = solve(remake(prob, p = Flux.data(p)), Tsit5(), saveat = 1.0)

    if save_curr == true
        plot(sol.t[:], sol[end, :], label = "Iteration #$training_cnt",
            color = :black,
            linestyle = :dash)
        plot!(test_data[:, 1], test_data[:, 2], label = "Test data",
            color = :black)
        plot!(fmt = :svg,
            xlims = (0, 200),
            ylims = (-200, 50),
            grid = false,
            legend = :topright)
        savefig("temperature_fit_$training_cnt.svg")

        df = DataFrame(x1 = Float64(test_data[1, 1]):Float64(test_data[end, 1]),
                       x2 = Flux.data(p)[1:end-1])
        write_checkpoint("checkpoint_htc_$training_cnt.csv", df)
        plot(Flux.data(p)[1:end-1], color = :black, label = "Iteration #$training_cnt")
        plot!(fmt = :svg,
            xlims = (0, 200),
            grid = false,
            legend = :topleft)
        savefig("checkpoint_htc_$training_cnt.svg")
    end

    if early_exit == true
        Flux.stop()
    end
end

cb()
Flux.train!(loss_rd, [p], data, opt, cb = cb)

# Save last iteration.
sol = solve(remake(prob, p = Flux.data(p)), Tsit5(), saveat = 1.0)
plot(sol.t[:], sol[end, :], label = "Iteration #$training_cnt",
    color = :black,
    linestyle = :dash)
plot!(test_data[:, 1], test_data[:, 2], label = "Test data",
    color = :black)
plot!(fmt = :svg,
    xlims = (0, 200),
    ylims = (-200, 50),
    grid = false,
    legend = :topright)
savefig("last_temperature_fit_$training_cnt.svg")

df = DataFrame(x1 = Float64(test_data[1, 1]):Float64(test_data[end, 1]),
            x2 = Flux.data(p)[1:end-1])
write_checkpoint("last_checkpoint_htc_$training_cnt.csv", df)
plot(Flux.data(p)[1:end-1], color = :black, label = "Iteration #$training_cnt")
plot!(fmt = :svg,
    xlims = (0, 200),
    grid = false,
    legend = :topleft)
savefig("last_checkpoint_htc_$training_cnt.svg")

# Save loss function.
df_loss = DataFrame(x1 = 1:length(tot_loss),
                    x2 = tot_loss)
write_checkpoint("loss.svg", df_loss)
plot(tot_loss, color = :black, label = "Loss",
    fmt = :svg,
    grid = false)
savefig("loss.svg")
