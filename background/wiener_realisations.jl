"""
Generate and plot realisations of 1- and 2-dimensional Wiener process, for
Chapter 2 (Background).
"""

using Random

using Distributions
using PyPlot

include("../pyplot_setup.jl")

Random.seed!(12402345)

begin
    #### Realisations of the Wiener process in 1D and 2D ####
    N = 5

    dt = 0.0001
    tspan = 0:dt:2

    # Make one figure!
    fig = figure(; figsize = figaspect(0.5))
    ax_1d = fig.add_subplot(121)
    ax_1d.set_xlabel(L"t")
    ax_1d.set_ylabel(L"W_t")

    ax_2d = fig.add_subplot(122)
    ax_2d.set_xlabel(L"W_t^{(1)}")
    ax_2d.set_ylabel(L"W_t^{(2)}")

    ax_2d.yaxis.tick_right()
    ax_2d.yaxis.set_label_position("right")

    d = Normal(0, sqrt(dt))

    for n in 1:N
        W1 = Vector{Float64}(undef, length(tspan))
        W2 = Array{Float64}(undef, 2, length(tspan))
        W1[1] = 0.0
        W2[:, 1] .= 0.0

        for i in 1:(length(tspan) - 1)
            W1[i + 1] = W1[i] + rand(d)
            W2[1, i + 1] = W2[1, i] + rand(d)
            W2[2, i + 1] = W2[2, i] + rand(d)
        end

        ax_1d.plot(tspan, W1; linewidth = 0.5, alpha = 0.7)

        # Only plot one realisation of the 2D process - too cluttered otherwise
        if n == 1
            ax_2d.plot(W2[1, :], W2[2, :], "k-"; linewidth = 0.5, alpha = 0.7)
        end
    end

    fig.subplots_adjust(; wspace = 0.1)

    fig.savefig("wiener_realisations.pdf"; bbox_inches = "tight", dpi = 600)
    close(fig)
end
