"""
Generate EM solution to the SDE
    dxₜ = sin(xₜ)dt + dWₜ
and plot for Chapter 2 (Background).
"""

using Random

using KernelDensity
using StochasticDiffEq
using PyPlot

include("../pyplot_setup.jl")

Random.seed!(19999999995567)

function u_sin(x, _, _)
    return sin(x)
end

function σ_sin(_, _, _)
    return 1.0
end

N = 100000
ts = 0:0.01:π
prob = SDEProblem(u_sin, σ_sin, 1.0, (0.0, π))
ens = EnsembleProblem(prob)
sol = solve(ens, EM(); trajectories = N, saveat = ts, dt = 0.01)
rels = [sol[i].u for i in 1:N]

# Solution without noise
Fs = @. 2 * acot(exp(-ts) * cot(1 / 2))

# Estimate the KDE
k = InterpKDE(kde(getindex.(rels, length(ts)); bandwidth = 0.5))

begin
    fig = figure(; figsize = figaspect(0.5))
    axs = fig.subplot_mosaic("AAB")
    ax1 = axs["A"]
    ax2 = axs["B"]

    # Some sample paths
    ax1.set_xlabel(L"t")
    ax1.set_ylabel(L"x_t")

    # Plot the first ten realisations
    for n in 1:5
        ax1.plot(ts, rels[n]; alpha = 0.7, linewidth = 0.5)
    end

    ax1.plot(ts, Fs; color = "black")

    # PDF at final time
    ax2.set_xlabel(L"x_\pi")
    ax2.set_ylabel(L"p")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    xgrid = range(-7; stop = 7, length = 10000)
    ax2.plot(xgrid, pdf.(Ref(k), xgrid), "k-")

    fig.subplots_adjust(; wspace = 0.1)
    fig.savefig("ou_solution.pdf"; bbox_inches = "tight", dpi = 600)
    close(fig)
end
