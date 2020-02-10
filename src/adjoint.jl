using CSV
using Flux
using Plots
using DataFrames
using DiffEqFlux
using DifferentialEquations

mutable struct hprob
    c # Specific heat.
    ρ # Density.
    k # Thermal conductivity.
    sinkT # Cryogenic liquid temperature.
    ambT # Initial condition.
    htcAmb # Atmospheric HTC.
end

# Calculate h as the interpolation for the current time point.
# TODO: precompute the value of 
function htc(h, ti)
    y1 = y2 = 0.0
    x1 = x2 = 0

    for (t, h) in enumerate(h)
        if ti >= t
            x1 = t
            y1 = h
        end
    end

    if x1 == length(h)
        return y1
    else
        x2 = x1 + 1
        y2 = h[x2]
        return (y2 - y1) / (x2 - x1) * (ti - x2) + y1
    end
end

# Discretise ODE.
# TODO: use StaticArrays for ODE state.
function heat_transfer(du, u, h, t)
    ΔT = (pr.sinkT + u[1]) / 2 - u[1]
    α = pr.k / (pr.ρ * pr.c)

    # Node exposed to cryogenic liquid.
    du[1] = 2 * (ΔT * htc(h, t) /
                 (pr.c * pr.ρ) - α * (u[1] - u[2]) / dx) / dx

    # Inner nodes.
    for i in 2:size(u)- 1
        du[i] = α * (u[i - 1] - 2u[i] + u[i + 1]) / dx^2
    end

    # Node exposed to ambient atmosphere. TC is located here.
    du[end] = 2 * (α * (u[end-1] - u[end]) /
                   dx - (u[end] - pr.ambT) * pr.htcAmb /
                   (pr.ρ * pr.c)) / dx
end

# Read test data.
function read_data()
    filename = "Fig_67.csv_Result.csv"
    data = CSV.read(filename)
    table = DataFrame(data)

    # Drop duplicates.
    nrows, ncols = size(table)
    for row in 2:nrows
        if table[row, 1] == table[row - 1, 1]
            println("Row ", row, " is a duplicate. Deleting...")
            table[row, 1] = 0
        end
    end

    table = table[table[1].!=0,:]

    return table
end

# Read saved htc values.
function read_checkpoint(filename)
    data = CSV.read(filename)
    table = DataFrame(data)

    return Vector(table[:, 2])
end

# Write htc values as checkpoints.
function write_checkpoint(filename, table)
    CSV.write(filename, table)
end

# Flux reverse-mode AD through the differential equation solver.
function predict_rd()
    diffeq_adjoint(p, prob, Tsit5(), saveat = 1.0)[end, :]
end

# Least squares error.
function loss_rd()
    sum(abs2, test_data[:, 2] .- predict_rd())
end

# Solve to get an idea of the initial guess.
function initial_solve(ini_h, u20, tspan, test_data)
    p = ini_h * ones(test_data[end, 1])
    prob = ODEProblem(heat_transfer, u20, tspan, p, saveat = 1.0)
    sol = solve(prob)
    plot(sol.t[:], sol[end, :], label = "Solution")
    plot!(test_data[:, 1], test_data[:, 2], label = "Test data")
end

test_data = read_data()

N = 10
t = 0.00635
dx = t / N

sinkT = test_data[end, 2]
ambT = test_data[1, 2]
u20 = ambT * ones(N)
tspan = (Float64(test_data[1, 1]), Float64(test_data[end, 1]))

pr = hprob(450.0, 7850.0, 42.0, sinkT, ambT, 4.0)

# Solve ODE.
p = param(read_checkpoint("initial_guess.csv"))

prob = ODEProblem(heat_transfer, u20, tspan, p, saveat = 1.0)

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
