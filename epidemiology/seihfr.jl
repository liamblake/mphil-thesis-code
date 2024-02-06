"""
Generate realisation and compute diffusion limit for the 5-dimensional Ebola model of Legrand et al.
(2007). This file generate the third and final figure in Section 7.7 of the thesis.
"""

using Distributions
using ProgressMeter
using PyPlot

using Random

include("../computations/gaussian_computation.jl")
include("../pyplot_setup.jl")

"""
    simulate_seihfr!(dest, N, βI, βH, βF, α, γh, γdh, γf, γi, θ1, δ1, δ2, γd, γih, M, I₀, T)

Generate N Monte-Carlo simulations of the SEIHFR CTMC model with infection rates βI, βH, βF, exposure
rate α, hospitalisation rate γh, death rate γd, removal rate γi, funeral rate γf, hospitalisation
death rate γdh, hospitalisation funeral rate γih, and funeral death rate γdh. The population size is
M, the initial condition is I₀ infected individuals, and the simulation is run up to time T. The
results are stored in dest, which is a 5×N matrix, where the first row is the number of susceptible
individuals, the second row is the number of exposed individuals, the third row is the number of
infected individuals, the fourth row is the number of hospitalised individuals, and the fifth row is
the number of dead individuals at the final time.

(Thanks Copilot for this unnecessarily verbose docstring)
"""
function simulate_seihfr!(dest, N, βI, βH, βF, α, γh, γdh, γf, γi, θ1, δ1, δ2, γd, γih, M, I₀, T)
    S = 0
    E = 0
    I = 0
    H = 0
    F = 0

    t = 0.0

    # Simulate N times
    for n in 1:N
        S = M - I₀
        E = 0
        I = I₀
        H = 0
        F = 0

        t = 0.0
        while t < T
            # Event rates
            r_expose = (βI * S * I + βH * S * H + βF * S * F) / M
            r_infect = α * E
            r_hospitalize = γh * θ1 * I
            r_death_hospital = γdh * δ2 * H
            r_burial = γf * F
            r_burial_infect = γi * (1 - θ1) * (1 - δ1) * I
            r_death_infect = δ1 * (1 - θ1) * γd * I
            r_burial_hospital = γih * (1 - δ2) * H

            # Sample time to next event
            total_rate =
                r_expose +
                r_infect +
                r_hospitalize +
                r_death_hospital +
                r_burial +
                r_burial_infect +
                r_death_infect +
                r_burial_hospital
            t += rand(Exponential(1 / total_rate))

            # Choose next event
            if t <= T
                r = rand() * total_rate
                if r < r_expose
                    # Exposure
                    S -= 1
                    E += 1
                elseif r < r_expose + r_infect
                    # Infect
                    E -= 1
                    I += 1
                elseif r < r_expose + r_infect + r_hospitalize
                    # Hospitalize
                    I -= 1
                    H += 1
                elseif r < r_expose + r_infect + r_hospitalize + r_death_hospital
                    # Death in hospital
                    H -= 1
                    F += 1
                elseif r < r_expose + r_infect + r_hospitalize + r_death_hospital + r_burial
                    # Burial
                    F -= 1
                elseif r <
                       r_expose +
                       r_infect +
                       r_hospitalize +
                       r_death_hospital +
                       r_burial +
                       r_burial_infect
                    # Removal of infected
                    I -= 1
                elseif r <
                       r_expose +
                       r_infect +
                       r_hospitalize +
                       r_death_hospital +
                       r_burial +
                       r_burial_infect +
                       r_death_infect
                    # Death of infected
                    I -= 1
                    F += 1
                else
                    # Burial of hospitalized
                    H -= 1
                end
            end
        end

        dest[:, n] = [S, E, I, H, F]
    end
end

######################################### MODEL PARAMETERS #########################################
I₀ = 20
T = 20.0
N = 10000
M = 20000
βI = 0.588
βH = 0.794
βF = 7.653
α = 1.0
γh = 7 / 5
γd = 35 / 48
γi = 7 / 10
γf = 7 / 2
γih = 1 / (1 / γi - 1 / γh)
γdh = 1 / (1 / γd - 1 / γh)

δ1 = 0.8
δ2 = 0.8
θ1 = 0.67

########################## GENERATE REALISATIONS AND COMPARE TO GAUSSIAN ###########################
sims = Array{Int64}(undef, 5, N)
simulate_seihfr!(sims, N, βI, βH, βF, α, γh, γdh, γf, γi, θ1, δ1, δ2, γd, γih, M, I₀, T)

sims_dens = sims ./ M

# Diffusion approximation and linearisation
function u!(s, x, _)
    s[1] = -βI * x[1] * x[3] - βH * x[1] * x[4] - βF * x[1] * x[5]
    s[2] = βI * x[1] * x[3] + βH * x[1] * x[4] + βF * x[1] * x[5] - α * x[2]
    s[3] = α * x[2] - (γh * θ1 + γi * (1 - θ1) * (1 - δ1) + δ1 * (1 - θ1) * γd) * x[3]
    s[4] = γh * θ1 * x[3] - (γdh * δ2 + γih * (1 - δ2)) * x[4]
    s[5] = γdh * δ2 * x[4] + δ1 * (1 - θ1) * γd * x[3] - γf * x[5]
end

function ∇u!(s, x, _)
    s .= 0.0

    s[1, 1] = -βI * x[3] - βH * x[4] - βF * x[5]
    s[1, 3] = -βI * x[1]
    s[1, 4] = -βH * x[1]
    s[1, 5] = -βF * x[1]

    s[2, 1] = βI * x[3] + βH * x[4] + βF * x[5]
    s[2, 2] = -α
    s[2, 3] = βI * x[1]
    s[2, 4] = βH * x[1]
    s[2, 5] = βF * x[1]

    s[3, 2] = α
    s[3, 3] = -γh * θ1 - γi * (1 - θ1) * (1 - δ1) - δ1 * (1 - θ1) * γd

    s[4, 3] = γh * θ1
    s[4, 4] = -γdh * δ2 - γih * (1 - δ2)

    s[5, 3] = δ1 * (1 - θ1) * γd
    s[5, 4] = γdh * δ2
    s[5, 5] = -γf
end

function σσᵀ!(s, x, _)
    s[1, 1] = βI * x[1] * x[3] + βH * x[1] * x[4] + βF * x[1] * x[5]
    s[1, 2] = -βI * x[1] * x[3] - βH * x[1] * x[4] - βF * x[1] * x[5]
    s[1, 3] = 0.0
    s[1, 4] = 0.0
    s[1, 5] = 0.0

    s[2, 1] = s[1, 2]
    s[2, 2] = βI * x[1] * x[3] + βH * x[1] * x[4] + βF * x[1] * x[5] + α * x[2]
    s[2, 3] = -α * x[2]
    s[2, 4] = 0.0
    s[2, 5] = 0.0

    s[3, 1] = s[1, 3]
    s[3, 2] = s[2, 3]
    s[3, 3] =
        α * x[2] + γh * θ1 * x[3] + γi * (1 - θ1) * (1 - δ1) * x[3] + δ1 * (1 - θ1) * γd * x[3]
    s[3, 4] = -γh * θ1 * x[3]
    s[3, 5] = -γdh * δ2 * x[4]

    s[4, 1] = s[1, 4]
    s[4, 2] = s[2, 4]
    s[4, 3] = s[3, 4]
    s[4, 4] = γh * θ1 * x[3] + γdh * δ2 * x[4] + γih * (1 - δ2) * x[4]
    s[4, 5] = -γdh * δ2 * x[4]

    s[5, 1] = s[1, 5]
    s[5, 2] = s[2, 5]
    s[5, 3] = s[3, 5]
    s[5, 4] = s[4, 5]
    s[5, 5] = γdh * δ2 * x[4] + γih * (1 - δ2) * x[4] + γf * x[5]
end

ts = range(0.0; stop = T, length = 1000)
F = Vector{Vector{Float64}}(undef, length(ts))
Σ = Vector{Matrix{Float64}}(undef, length(ts))

# Compute Gaussian process approximation
x0 = [1 - I₀ / M, 0.0, I₀ / M, 0.0, 0.0]
gaussian_computation!(F, Σ, 5, u!, ∇u!, σσᵀ!, x0, zeros(5, 5), ts)

function invisible_ax!(ax)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.spines["left"].set_visible(false)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
end

# Big old pairwise plot
fdir = "../../thesis/chp07_outlook/figures/seihfr"
xgrid = 0:0.0001:1
begin
    fig = figure(; figsize = figaspect(0.5))

    # Marginal histograms
    ax11 = fig.add_subplot(5, 5, 1)
    ax11.hist(sims_dens[1, :]; bins = 50, color = :green, alpha = 0.8, density = true)
    ax11.set_xlabel(L"S/M"; fontsize = 12)

    ax22 = fig.add_subplot(5, 5, 7)
    ax22.hist(sims_dens[2, :]; bins = 50, color = :blue, alpha = 0.8, density = true)
    ax22.set_xlabel(L"E/M"; fontsize = 12)

    ax33 = fig.add_subplot(5, 5, 13)
    ax33.hist(sims_dens[3, :]; bins = 50, color = :red, alpha = 0.8, density = true)
    ax33.set_xlabel(L"I/M"; fontsize = 12)

    ax44 = fig.add_subplot(5, 5, 19)
    ax44.hist(sims_dens[4, :]; bins = 50, color = :yellow, alpha = 0.8, density = true)
    ax44.set_xlabel(L"H/M"; fontsize = 12)

    ax55 = fig.add_subplot(5, 5, 25)
    ax55.hist(sims_dens[5, :]; bins = 50, color = :orange, alpha = 0.8, density = true)
    ax55.set_xlabel(L"D/M"; fontsize = 12)
    ax55.set_yticks([])

    # Pairs
    ax12 = fig.add_subplot(5, 5, 2; sharex = ax22)
    ax12.hist2d(sims_dens[2, :], sims_dens[1, :]; bins = 50, cmap = :Purples, rasterized = true)
    # ax12.set_title(L"E/M"; fontsize = 12)

    ax13 = fig.add_subplot(5, 5, 3; sharex = ax33)
    ax13.hist2d(sims_dens[3, :], sims_dens[1, :]; bins = 50, cmap = :Purples, rasterized = true)
    # ax13.set_title(L"I/M"; fontsize = 12)

    ax14 = fig.add_subplot(5, 5, 4; sharex = ax44)
    ax14.hist2d(sims_dens[4, :], sims_dens[1, :]; bins = 50, cmap = :Purples, rasterized = true)
    # ax14.set_title(L"H/M"; fontsize = 12)

    ax15 = fig.add_subplot(5, 5, 5; sharex = ax55)
    ax15.hist2d(sims_dens[5, :], sims_dens[1, :]; bins = 50, cmap = :Purples, rasterized = true)
    ax15.set_ylabel(L"S/M"; fontsize = 12)
    # ax15.set_title(L"D/M"; fontsize = 12)

    ax23 = fig.add_subplot(5, 5, 8; sharex = ax33)
    ax23.hist2d(sims_dens[3, :], sims_dens[2, :]; bins = 50, cmap = :Purples, rasterized = true)

    ax24 = fig.add_subplot(5, 5, 9; sharex = ax44)
    ax24.hist2d(sims_dens[4, :], sims_dens[2, :]; bins = 50, cmap = :Purples, rasterized = true)

    ax25 = fig.add_subplot(5, 5, 10; sharex = ax55)
    ax25.hist2d(sims_dens[5, :], sims_dens[2, :]; bins = 50, cmap = :Purples, rasterized = true)
    ax25.set_ylabel(L"E/M"; fontsize = 12)

    ax34 = fig.add_subplot(5, 5, 14; sharex = ax44)
    ax34.hist2d(sims_dens[4, :], sims_dens[3, :]; bins = 50, cmap = :Purples, rasterized = true)

    ax35 = fig.add_subplot(5, 5, 15; sharex = ax55)
    ax35.hist2d(sims_dens[5, :], sims_dens[3, :]; bins = 50, cmap = :Purples, rasterized = true)
    ax35.set_ylabel(L"I/M"; fontsize = 12)

    ax45 = fig.add_subplot(5, 5, 20; sharex = ax55)
    ax45.hist2d(sims_dens[5, :], sims_dens[4, :]; bins = 50, cmap = :Purples, rasterized = true)
    ax45.set_ylabel(L"H/M"; fontsize = 12)

    # Reduce space between plots
    fig.subplots_adjust(; hspace = 0.075, wspace = 0.075)
    fig.align_ylabels()

    # Set ticks and labels to right
    ax15.yaxis.tick_right()
    ax15.yaxis.set_label_position("right")
    ax25.yaxis.tick_right()
    ax25.yaxis.set_label_position("right")
    ax35.yaxis.tick_right()
    ax35.yaxis.set_label_position("right")
    ax45.yaxis.tick_right()
    ax45.yaxis.set_label_position("right")
    ax55.yaxis.tick_right()
    ax55.yaxis.set_label_position("right")

    # Hide interor labels
    for i in 1:5
        for j in i:5
            # row i, column j
            ax = eval(Symbol("ax$i$j"))

            if j == 5
                # Final column
                ax.tick_params(; axis = "y", labelsize = 8)
            else
                ax.tick_params(; axis = "y", labelleft = false, length = 0)
            end
            if i == j
                # Diagonals
                ax.tick_params(; axis = "x", labelsize = 8)
            else
                ax.tick_params(; axis = "x", labelbottom = false, length = 0)
            end
        end
    end

    fig.savefig("$fdir/seihfr_marginals.pdf"; bbox_inches = "tight")

    # Now add the Gaussian approximations
    for i in 1:5
        for j in i:5
            ax = eval(Symbol("ax$i$j"))
            if i == j
                # PDF
                ax.plot(
                    xgrid,
                    pdf.(Normal(F[end][i], sqrt(Σ[end][i, i] / M)), xgrid);
                    color = :black,
                    linewidth = 0.5,
                    scalex = false,
                )
            else
                # Covariance ellipses
                Π = [Σ[end][j, j] Σ[end][j, i]; Σ[end][i, j] Σ[end][i, i]]
                vals, evecs = eigen(1 / M * Π)
                θ = atand(evecs[2, 2], evecs[1, 2])
                ell_w = sqrt(vals[2])
                ell_h = sqrt(vals[1])
                for l in [1, 2]
                    ax.add_artist(
                        PyPlot.matplotlib.patches.Ellipse(;
                            xy = F[end][[j, i]],
                            width = 2 * l * ell_w,
                            height = 2 * l * ell_h,
                            angle = θ,
                            edgecolor = :red,
                            facecolor = :none,
                            linewidth = 0.5,
                            linestyle = "solid",
                            zorder = 2,
                        ),
                    )
                end
            end
        end
    end

    fig.savefig("$fdir/seihfr_marginals_gaussian.pdf"; bbox_inches = "tight")

    close(fig)
end
