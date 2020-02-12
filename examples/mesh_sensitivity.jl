# Determine the mesh sensitivity for the default unexposed HTC and the recovered value of
# the exposed HTC. This is ran post-factum of the recovery calculation to justify it.
# In case the correct mesh was not picked the adjoint calculation muts be repeated with
# the converged value of the mesh size.

using NeuralThermalODE

const testfile = "Fig_67.csv_Result.csv"
const test_data = read_data(testfile)

# This is either the raw HTC or a spline fit.
const htc_file = "checkpoint_htc_360.csv"
const htc_recovered = read_htc(htc_file)

const sinkT = test_data[end, 2]
const ambT = test_data[1, 2]

tspan = (Float64(test_data[1, 1]), Float64(test_data[end, 1]))

const p = Vector(htc_recovered[2])

# Pick correct value.
const t = 0.00635

# Study the mesh sensitivity.
const N1 = 10
const N2 = 20
const N3 = 40
const N4 = 80

dx = t / N1
u = [ambT for i in 1:N1]
prob = ODEProblem(heat_transfer, u, tspan, p, saveat = 1.0)
sol1 = solve(prob, Tsit5(), saveat = 1.0)

dx = t / N2
u = [ambT for i in 1:N2]
prob = ODEProblem(heat_transfer, u, tspan, p, saveat = 1.0)
sol2 = solve(prob, Tsit5(), saveat = 1.0)

dx = t / N3
u = [ambT for i in 1:N3]
prob = ODEProblem(heat_transfer, u, tspan, p, saveat = 1.0)
sol3 = solve(prob, Tsit5(), saveat = 1.0)

dx = t / N4
u = [ambT for i in 1:N4]
prob = ODEProblem(heat_transfer, u, tspan, p, saveat = 1.0)
sol4 = solve(prob, Tsit5(), saveat = 1.0)

plot(sol1.t[:], sol1[end, :], label = "N = 10", color = :black, linestyle = :dash)
plot!(sol2.t[:], sol2[end, :], label = "N = 20", color = :black, linestyle = :dot)
plot!(sol3.t[:], sol3[end, :], label = "N = 40", color = :black, linestyle = :dashdot)
plot!(sol4.t[:], sol4[end, :], label = "N = 80", color = :black)
plot!(fmt = :svg,
      grid = false,
      xlabel = "Time (min)",
      ylabel = "Temperature (°C)",
      xlims = (0, 200),
      ylims = (-200, 50),
      xticks = 0:25:200,
      yticks = -200:25:50,
      minorticks = true)

savefig("mesh_sensitivity.svg")
