"""
Apply stochastic sensitivity results to the SIR CTMC model. This file generate the first two figures
in Section 7.7 of the thesis.
"""

using Distributions
using ProgressMeter
using PyPlot

include("../computations/gaussian_computation.jl")
include("../pyplot_setup.jl")
save_dpi = 600

"""
    simulate_sir!(dest, N, β, γ, M, I₀, T)

Generate N Monte-Carlo simulations of the SIR CTMC model with infection rate β, recovery rate γ, and
population size M. The initial condition is I₀ infected individuals, and the simulation is run up to
time T. The results are stored in dest, which is a 2×N matrix, where the first row is the number of
susceptible individuals and the second row is the number of infected individuals at the final time.
"""
function simulate_sir!(dest, N, β, γ, M, I₀, T)
    S = 0
    I = 0
    t = 0.0

    # Simulate N times
    for n in 1:N
        I = I₀
        S = M - I₀
        t = 0.0
        while t < T
            # Event rates
            r_infect = β * I * S / M
            r_recover = γ * I

            # Sample time to next event
            t += rand(Exponential(1 / (r_infect + r_recover)))

            # Choose next event
            if t <= T
                if rand() < (r_infect) / (r_infect + r_recover)
                    # Infect
                    S -= 1
                    I += 1
                else
                    # Recover
                    I -= 1
                end
            end
        end

        dest[:, n] = [S, I]
    end
end


# Diffusion approximation and linearisation
function u!(s, x, _)
    s[1] = -β * x[1] * x[2]
    s[2] = β * x[1] * x[2] - γ * x[2]
end

function ∇u!(s, x, _)
    s[1, 1] = -β * x[2]
    s[1, 2] = -β * x[1]
    s[2, 1] = β * x[2]
    s[2, 2] = β * x[1] - γ
end

function σ!(s, x, _)
    s[1, 1] = sqrt(β * x[1] * x[2])
    s[1, 2] = 0.0
    s[2, 1] = -sqrt(β * x[1] * x[2])
    s[2, 2] = sqrt(γ * x[2])
end

function σσᵀ!(s, x, _)
    s[1, 1] = β * x[1] * x[2]
    s[1, 2] = -β * x[1] * x[2]
    s[2, 1] = s[1, 2]
    s[2, 2] = β * x[1] * x[2] + γ * x[2]
end

# Output directory for figures
fdir = "../../thesis/chp07_outlook/figures/sir"

########################## GENERATE REALIASATIONS AND COMPARE TO GAUSSIAN ##########################
T = 5
N = 100000
sims = Array{Int64}(undef, 2, N)
β = 1.2
γ = 0.8
dt = 0.01
ts = 0.0:dt:T

F = Vector{Vector{Float64}}(undef, length(ts))
Σ = Vector{Matrix{Float64}}(undef, length(ts))

@showprogress for M in [50, 100, 1000, 10000]
    I₀ = 0.1 * M
    simulate_sir!(sims, N, β, γ, M, I₀, T)

    dens_sims = sims ./ M

    # Number of bins in each dimension
    Mm = min(M)#, 1000)
    s_bins = range(minimum(dens_sims[1, :]); stop=maximum(dens_sims[1, :]), step=1 / Mm)
    i_bins = range(minimum(dens_sims[2, :]); stop=maximum(dens_sims[2, :]), step=1 / Mm)

    # Compute Gaussian process approximation
    x0 = [M - I₀, I₀] ./ M
    gaussian_computation!(F, Σ, 2, u!, ∇u!, σσᵀ!, x0, zeros(2, 2), ts)

    fig = figure()
    axs = fig.subplot_mosaic("""
            AAA.
            BBBC
            BBBC
            BBBC
        """)

    ax_joint = axs["B"]
    ax_joint.set_xlabel(L"S/M")
    ax_joint.set_ylabel(L"I/M")

    h, xedge, yedge, _ = ax_joint.hist2d(
        dens_sims[1, :],
        dens_sims[2, :];
        bins=[s_bins, i_bins],
        density=true,
        cmap=:Purples,
        rasterized=true,
    )

    ax_joint.scatter(F[end][1], F[end][2]; color=:red, s=5)

    vals, evecs = eigen(1 / M * Σ[end])
    θ = atand(evecs[2, 2], evecs[1, 2])
    ell_w = sqrt(vals[2])
    ell_h = sqrt(vals[1])
    for l in [1, 2, 3]
        ax_joint.add_artist(
            PyPlot.matplotlib.patches.Ellipse(;
                xy=F[end],
                width=2 * l * ell_w,
                height=2 * l * ell_h,
                angle=θ,
                edgecolor=:red,
                facecolor=:none,
                linewidth=1.0,
                linestyle="solid",
                zorder=2,
            ),
        )
    end

    ax_S = axs["A"]
    ax_S.axis("off")
    ax_S.hist(
        dens_sims[1, :];
        bins=xedge,
        align="mid",
        density=true,
        color=(120.0 / 255.0, 115.0 / 255.0, 175.0 / 255.0),
    )
    sgrid = minimum(dens_sims[1, :]):(1/10000):maximum(dens_sims[1, :])
    ax_S.plot(
        sgrid, # Adjusted for bin centering
        pdf.(Normal(F[end][1], sqrt(1 / M * Σ[end][1, 1])), sgrid);
        color=:red,
        linewidth=1.0,
    )
    ax_S.set_xlim(xedge[1], xedge[end])

    ax_I = axs["C"]
    ax_I.axis("off")
    ax_I.hist(
        dens_sims[2, :];
        bins=yedge,
        align="mid",
        density=true,
        color=(120.0 / 255.0, 115.0 / 255.0, 175.0 / 255.0),
        orientation="horizontal",
    )
    igrid = minimum(dens_sims[2, :]):(1/10000):maximum(dens_sims[2, :])
    ax_I.plot(
        pdf.(Normal(F[end][2], sqrt(1 / M * Σ[end][2, 2])), igrid),
        igrid;
        color=:red,
        linewidth=1.0,
    )
    ax_I.set_ylim(yedge[1], yedge[end])

    fig.subplots_adjust(; wspace=0, hspace=0)
    fig.savefig("$fdir/sir_pairwise_$(M).pdf"; bbox_inches="tight", dpi=save_dpi)
    close(fig)

end


############################## PLOT STOCHASTIC SENSITIVITY VERSUS R₀ ###############################
R0s = 1 ./ (0.1:0.01:3)
i = 0.1

S2s = Vector{Float64}(undef, length(R0s))

N = 10000
sims = Array{Int64}(undef, 2, N)
Ms = [500, 1000, 5000, 10000, 50000]
sim_vars = Array{Float64}(undef, length(Ms), length(R0s))

β = 1
@showprogress for (j, R0) in enumerate(R0s)
    γ = β / R0

    function u!(s, x, _)
        s[1] = -β * x[1] * x[2]
        s[2] = β * x[1] * x[2] - γ * x[2]
    end

    function ∇u!(s, x, _)
        s[1, 1] = -β * x[2]
        s[1, 2] = -β * x[1]
        s[2, 1] = β * x[2]
        s[2, 2] = β * x[1] - γ
    end

    function σσᵀ!(s, x, _)
        s[1, 1] = β * x[1] * x[2]
        s[1, 2] = -β * x[1] * x[2]
        s[2, 1] = s[1, 2]
        s[2, 2] = β * x[1] * x[2] + γ * x[2]
    end

    gaussian_computation!(F, Σ, 2, u!, ∇u!, σσᵀ!, [1 - i, i], zeros(2, 2), ts)
    S2s[j] = opnorm(Σ[end])

    # Generate many simulations (with different population sizes)
    for (k, M) in enumerate(Ms)
        simulate_sir!(sims, N, β, γ, M, ceil(i * M), T)
        sim_vars[k, j] = opnorm(var(sims; dims=2)) ./ M
    end
end

# Make a plot
begin
    fig, ax = subplots(; figsize=figaspect(0.5))
    ax.set_xlabel(L"\gamma")
    ax.set_ylabel("Variance")
    for (k, M) in enumerate(Ms)
        ax.plot(1 ./ R0s, sim_vars[k, :]; alpha=0.5, label=L"M = %$M", zorder=2)
    end

    ax.plot(1 ./ R0s, S2s; color=:black, linewidth=2.0, label=L"S^2", zorder=1)

    # ax.set_yscale("log")
    ax.legend()

    fig.savefig("$fdir/sir_s2_R0.pdf"; bbox_inches="tight", dpi=save_dpi)

    ax.set_yscale("log")
    fig.savefig("$fdir/sir_s2_R0_log.pdf"; bbox_inches="tight", dpi=save_dpi)

    close(fig)