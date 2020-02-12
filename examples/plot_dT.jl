using NeuralThermalODE

# This is either the raw HTC or a spline fit.
htc_file = "checkpoint_htc_360.csv"
htc_recovered = read_htc(htc_file)

# Experimental LN2 HTC
exp_file = "ln2.csv"
exp_recovered = read_htc(exp_file)

const testfile = "Fig_67.csv_Result.csv"
const test_data = read_data(testfile)

const N = 10
const t = 0.00635
const dx = t / N

const sinkT = test_data[end, 2]
const ambT = test_data[1, 2]
const u20 = [ambT for i in 1:N]

tspan = (Float64(test_data[1, 1]), Float64(test_data[end, 1]))

# Solve ODE.
const p = Vector(htc_recovered[2])

prob = ODEProblem(heat_transfer, u20, tspan, p, saveat = 1.0)
sol = solve(prob, Tsit5(), saveat = 1.0)
#plot(sol.t[:], sol[1, :], label = "Exposed")
#plot!(sol.t[:], sol[end, :], label = "Unexposed")

exposedT = sol[1, :]
vapourT = (exposedT .+ sinkT) / 2
ΔT = abs.(sinkT .- vapourT)
plot(xlims = (1, 120))
#plot!(xscale = :log, yscale = :log)
plot!(xlabel = "\\Delta T", ylabel = "HTC W/mK")
plot!(ΔT, p, label = "Recovered")
plot!(exp_recovered[1], exp_recovered[2], label = "Small scale experiment")
