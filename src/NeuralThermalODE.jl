module NeuralThermalODE

using Reexport

@reexport using CSV
@reexport using Flux
@reexport using Plots
@reexport using DataFrames
@reexport using DiffEqFlux
@reexport using DifferentialEquations

struct ThermalODE{T}
    c::T        # Specific heat.
    ρ::T        # Density.
    k::T        # Thermal conductivity.
    sinkT::T    # Cryogenic liquid temperature.
    ambT::T     # Initial condition.
    htcAmb:: T  # Atmospheric HTC.
end

function BuildThermalODE(c, ρ, k, sinkT, ambT, htcAmb)
    ThermalODE(c, ρ, k, sinkT, ambT, htcAmb)
end

# Calculate h as the interpolation for the current time point.
# TODO: precompute this. Not point to iterate over the whole array and access can be done
# in O(1).
function htc(h, ti)
    y1 = y2 = 0.0
    x1 = x2 = 0

    for (t, h) in enumerate(h)
        if ti >= t
            x1 = t
            y1 = h
        end
    end

    if x1 == size(h)
        return y1
    else
        x2 = x1 + 1
        y2 = h[x2]
        return (y2 - y1) / (x2 - x1) * (ti - x2) + y1
    end
end

# Discretise ODE.
function heat_transfer(pr, du, u, h, t)
    ΔT = (pr.sinkT + u[1]) / 2 - u[1]
    α = pr.k / (pr.ρ * pr.c)

    # Node exposed to cryogenic liquid.
    du[1] = 2 * (ΔT * htc(h, t) /
                 (pr.c * pr.ρ) - α * (u[1] - u[2]) / dx) / dx

    # Inner nodes.
    for i in 2:size(u) - 1
        du[i] = α * (u[i - 1] - 2u[i] + u[i + 1]) / dx^2
    end

    # Node exposed to ambient atmosphere. TC is located here.
    du[end] = 2 * (α * (u[end-1] - u[end]) /
                   dx - (u[end] - pr.ambT) * pr.htcAmb /
                   (pr.ρ * pr.c)) / dx
end

# Read test data.
function read_data(filename)
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

# Read saved htc values and return as parameters to be optimized.
function read_checkpoint(filename)
    data = CSV.read(filename)
    table = DataFrame(data)

    return param(Vector(table[:, 2]))
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

export BuildThermalODE,
       read_checkpoint,
       write_checkpoint,
       read_data,
       predict_rd,
       loss_rd,
       initial_solve

end
