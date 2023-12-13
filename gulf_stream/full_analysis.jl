"""
A complete analysis of a trajectory in the Gulf Stream, using an SDE model constructed from
altimetry-derived velocity data. This file generates all results and figures for Section 6.2 of the
thesis.

Note that this script relies upon a NetCDF file containing the pre-processed altimetry data, the
path of which is specified as `nc_data`. This file is not included in the repository due to its
size, but can be downloaded from the Copernicus Marine Environment Monitoring Service (CMEMS) at
    https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description

"""

using Dates
using LinearAlgebra
using Random

using ColorSchemes
using DifferentialEquations
using Distances
using Distributions
using Interpolations
using ProgressMeter
using PyPlot
using StatsBase

include("../computations/gaussian_computation.jl")
include("../computations/hellinger.jl")
include("../computations/sigma_points.jl")
include("process_ocean.jl")

include("../pyplot_setup.jl")

Random.seed!(1328472345)

# Signed square root
ssqrt(x) = sign(x) * sqrt(abs(x))

#################################### LOAD DATA AND SETUP MODELS ####################################
# Load data
nc_data = "data/north_atlantic_gulf/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_1680670084978.nc"
lon, lat, days, dates, u, v, v_err, u_err, ssh, land, dx, _ = parse_glo_phy_l4_daucs(nc_data);

# Construct interpolations
u_interp = linear_interpolation((lon, lat, days), u; extrapolation_bc = 0.0)
v_interp = linear_interpolation((lon, lat, days), v; extrapolation_bc = 0.0)
u_err_interp = linear_interpolation((lon, lat, days), u_err; extrapolation_bc = 0.0)
v_err_interp = linear_interpolation((lon, lat, days), v_err; extrapolation_bc = 0.0)
# Meshgrid-style arrangement of longitude and latitude values - useful for contour maps
glon = repeat(lon, 1, length(lat))
glat = repeat(lat; inner = (1, length(lon)))'

function vel!(s, x, t)
    s[1] = u_interp(x[1], x[2], t)
    s[2] = v_interp(x[1], x[2], t)
    nothing
end

# Finite-difference approximation of ∇u
function ∇u!(s, x, t)
    s[1, 1] = u_interp(x[1] + dx, x[2], t) - u_interp(x[1] - dx, x[2], t)
    s[1, 2] = u_interp(x[1], x[2] + dx, t) - u_interp(x[1], x[2] - dx, t)
    s[2, 1] = v_interp(x[1] + dx, x[2], t) - v_interp(x[1] - dx, x[2], t)
    s[2, 2] = v_interp(x[1], x[2] + dx, t) - v_interp(x[1], x[2] - dx, t)
    rmul!(s, 1 / (2 * dx))
    nothing
end

# Interpolated ssh
ssh_interp = linear_interpolation((lon, lat, days), ssh)

# Diffusion - from data
ε = sqrt(dx)
function σ!(s, x, t)
    s[1, 1] = ssqrt(u_err_interp(x[1], x[2], t))
    s[2, 2] = ssqrt(v_err_interp(x[1], x[2], t))
    s[1, 2] = 0.0
    s[2, 1] = 0.0
    rmul!(s, ε)
    nothing
end

function σσᵀ!(s, x, t)
    s[1, 1] = abs(u_err_interp(x[1], x[2], t))
    s[2, 2] = abs(v_err_interp(x[1], x[2], t))
    s[1, 2] = 0.0
    s[2, 1] = 0.0
    rmul!(s, ε^2)
    nothing
end

model = SDEModel(2, vel!, ∇u!, σ!, σσᵀ!)

# Window for plotting
lon_range = (-66, -52)
lat_range = (34, 46)
biglon = minimum(lon_range):0.01:maximum(lon_range)
biglat = minimum(lat_range):0.01:maximum(lat_range)

# Output settings
fdir = "../../thesis/chp06_applications/figures/gulf_stream"
dpi = 600

# Timeframe details
i1 = 1
i2 = 8
t1 = days[i1]
t2 = days[i2]
dt = 1 / (24)
tspan = t1:dt:t2

###################################### PLOTS OF EULERIAN DATA ######################################
abs_speed = @. sqrt(u^2 + v^2)

# Still of SSH for introduction diagram. Very contrived, no need to be precise.
begin
    fig, ax = subplots()

    sshnonan = copy(ssh)
    sshnonan[isnan.(sshnonan)] .= 0.0

    ssh_interp2 = cubic_spline_interpolation(
        (minimum(lon):0.25:maximum(lon), minimum(lat):0.25:maximum(lat), 0:1:(length(days) - 1)),
        sshnonan,
    )

    # Kill the axis
    for spine in ax.spines.values()
        spine.set_visible(false)
    end
    ax.set_xticks([])
    ax.set_yticks([])

    ax.contour(
        biglon,
        biglat,
        ssh_interp2.(
            repeat(biglon, 1, length(biglat)),
            repeat(biglat; inner = (1, length(biglon)))',
            20.0,
        )';
        colors = "grey",
        levels = 20,
        linewidths = 0.75,
        zorder = 0,
        negative_linestyles = "solid",
    )

    xlim((-62, -56))
    ylim((36, 43))

    fig.savefig(
        "../../thesis/chp01_introduction/figures/gulf_stream_ssh.pdf";
        transparent = true,
        dpi = dpi,
        bbox_inches = "tight",
    )
end

# Contour plots at various times
for i in range(1; step = 5, length = 4)
    println("Saving timestamp $(dates[i]) to streamlines_$(i - 1).pdf")

    fig, ax = subplots()
    ax.set_xlabel("°W")
    ax.set_ylabel("°N")

    cf = ax.contourf(lon, lat, abs_speed[:, :, i]'; cmap = "thermal", levels = 20)
    ax.contour(
        lon,
        lat,
        ssh[:, :, i]';
        colors = "grey",
        linewidths = 0.75,
        levels = 15,
        negative_linestyles = "solid",
    )

    # Overlay land
    ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

    xlim(lon_range)
    ylim(lat_range)

    ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))

    colorbar(
        cf;
        ax = ax,
        location = "top",
        aspect = 40,
        label = L"Speed ($\mathrm{degrees}/\mathrm{day}$)",
    )

    fig.savefig("$fdir/streamlines_$(i-1).pdf"; dpi = dpi, bbox_inches = "tight")
    close(fig)

    # Also take a look at the error in the meridonal direction
    fig, ax = subplots()
    ax.set_xlabel("°W")
    ax.set_ylabel("°N")

    cf = ax.pcolormesh(lon, lat, u_err[:, :, i]'; cmap = "thermal", rasterized = true)
    ax.contour(
        lon,
        lat,
        ssh[:, :, i]';
        colors = "grey",
        linewidths = 0.75,
        levels = 15,
        negative_linestyles = "solid",
    )

    # Overylay land
    ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

    colorbar(
        cf;
        ax = ax,
        location = "top",
        aspect = 40,
        label = L"Error ($\mathrm{degrees}/\mathrm{day}$)",
    )

    xlim(lon_range)
    ylim(lat_range)

    ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))

    fig.savefig("$fdir/u_err_$(i-1).pdf"; dpi = dpi, bbox_inches = "tight")
    close(fig)

    fig, ax = subplots()
    ax.set_xlabel("°W")
    ax.set_ylabel("°N")

    cf = ax.pcolormesh(lon, lat, v_err[:, :, i]'; cmap = "thermal", rasterized = true)
    ax.contour(
        lon,
        lat,
        ssh[:, :, i]';
        colors = "grey",
        linewidths = 0.75,
        levels = 15,
        negative_linestyles = "solid",
    )

    # Overylay land
    ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

    colorbar(
        cf;
        ax = ax,
        location = "top",
        aspect = 40,
        label = L"Error ($\mathrm{degrees}/\mathrm{day}$)",
    )

    xlim(lon_range)
    ylim(lat_range)

    ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))

    fig.savefig("$fdir/v_err_$(i-1).pdf"; dpi = dpi, bbox_inches = "tight")
    close(fig)
end

################################# ANALYSIS OF A SINGLE TRAJECTORY ##################################
# Initial condition
x₀ = [-60.5, 39.0]

# Compute the Gaussian approximation about the trajectory
ws = Vector{Vector}(undef, length(tspan))
Σs = Vector{Matrix}(undef, length(tspan))
gaussian_computation!(ws, Σs, 2, vel!, ∇u!, σσᵀ!, x₀, zeros(2, 2), tspan)

# Plot the trajectory through time
begin
    fig = figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.contour(
        lon,
        lat,
        ssh_interp.(glon, glat, days[i2])';
        colors = "grey",
        levels = 20,
        linewidths = 0.75,
        zorder = 0,
        negative_linestyles = "solid",
    )
    ax.plot(getindex.(ws, 1), getindex.(ws, 2), "k-"; linewidth = 1.5)
    ax.scatter(ws[end][1], ws[end][2]; c = "blue", s = 2.5, zorder = 5)

    # Overlay land
    ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

    xlim(lon_range)
    ylim(lat_range)
    ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
    ax.set_xlabel("°W")
    ax.set_ylabel("°N")

    savefig("$fdir/det_traj.pdf"; dpi = dpi, bbox_inches = "tight")

    close(fig)
end

# Generate realisations of the corresponding stochastic trajectory
# Drift and diffusion need to be written in the form required by the SDEProblem
function sciml_u!(ds, x, _, t)
    vel!(ds, x, t)
end

function sciml_σ!(ds, x, _, t)
    σ!(ds, x, t)
    # rmul!(ds, ε)
end

# Setup the SDEProblem and generate the realisations
N = 10000
sde_prob =
    SDEProblem(sciml_u!, sciml_σ!, x₀, (days[i1], days[i2]); noise_rate_prototype = zeros(2, 2))
ens = EnsembleProblem(sde_prob)
num_sol = solve(ens, EM(), EnsembleThreads(); trajectories = N, dt = dt, saveat = tspan)
num_rels = Array(num_sol)

# Scatter plot of realisations
begin
    fig, ax = subplots()

    ax.contour(
        lon,
        lat,
        ssh_interp.(glon, glat, days[i2])';
        colors = "grey",
        levels = 20,
        zorder = 0,
        linewidths = 0.75,
        negative_linestyles = "solid",
    )
    ax.scatter(ws[end][1], ws[end][2]; c = "blue", s = 2.5, zorder = 5)

    # Samples
    ax.scatter(
        num_rels[1, end, 1:2500],
        num_rels[2, end, 1:2500];
        c = "red",
        s = 1.0,
        alpha = 0.6,
        zorder = 2,
    )

    # Overlay land
    ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

    xlim(lon_range)
    ylim(lat_range)
    ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
    ax.set_xlabel("°W")
    ax.set_ylabel("°N")

    savefig("$fdir/traj_stoch_rels.pdf"; dpi = dpi, bbox_inches = "tight")
    close(fig)
end

# Histogram of samples
begin
    fig = figure()
    ax_joint = fig.add_subplot(2, 2, 3)
    ax_joint.set_xlabel("°W")
    ax_joint.set_ylabel("°N")

    ax_joint.set_facecolor((0.9882352941176471, 0.9882352941176471, 0.9921568627450981))

    h, xedge, yedge, _ = ax_joint.hist2d(
        num_rels[1, end, :],
        num_rels[2, end, :];
        bins = 100,
        density = true,
        cmap = :Purples,
        rasterized = true,
    )
    ax_joint.set_xticks(ax_joint.get_xticks(), -ax_joint.get_xticks())

    ax_joint.scatter(ws[end][1], ws[end][2]; color = :red, s = 5)

    ax_S = fig.add_subplot(2, 2, 1)
    ax_S.axis("off")
    ax_S.hist(
        num_rels[1, end, :];
        align = "left",
        bins = 50,
        density = true,
        color = (120.0 / 255.0, 115.0 / 255.0, 175.0 / 255.0),
    )

    ax_I = fig.add_subplot(2, 2, 4)
    ax_I.axis("off")
    ax_I.hist(
        num_rels[2, end, :];
        align = "left",
        bins = 50,
        density = true,
        color = (120.0 / 255.0, 115.0 / 255.0, 175.0 / 255.0),
        orientation = "horizontal",
    )

    fig.subplots_adjust(; wspace = 0, hspace = 0)
    fig.savefig("$fdir/traj_stoch_joint.pdf"; bbox_inches = "tight", dpi = dpi)

    # Overlay Gaussian
    vals, evecs = eigen(Σs[end])
    θ = atand(evecs[2, 2], evecs[1, 2])
    ell_w = sqrt(vals[2])
    ell_h = sqrt(vals[1])
    for l in [1, 2, 3]
        ax_joint.add_artist(
            PyPlot.matplotlib.patches.Ellipse(;
                xy = ws[end],
                width = 2 * l * ell_w,
                height = 2 * l * ell_h,
                angle = θ,
                edgecolor = :red,
                facecolor = :none,
                linewidth = 1.0,
                linestyle = "solid",
                zorder = 2,
            ),
        )
    end

    xgrid = range(minimum(num_rels[1, end, :]); stop = maximum(num_rels[1, end, :]), length = 1000)
    ygrid = range(minimum(num_rels[2, end, :]); stop = maximum(num_rels[2, end, :]), length = 1000)

    ax_S.plot(xgrid, pdf.(Normal(ws[end][1], sqrt(Σs[end][1, 1])), xgrid); color = :red)
    ax_I.plot(pdf.(Normal(ws[end][2], sqrt(Σs[end][2, 2])), ygrid), ygrid; color = :red)

    fig.savefig("$fdir/traj_stoch_joint_gauss.pdf"; bbox_inches = "tight", dpi = dpi)

    close(fig)
end

"""
    qqplot!(ax, x; reference_line=true)

Plot a normal quantile-quantile (QQ) plot of the data `x` on the PyPlot axis `ax`. If
`reference_line` is true, a dashed red line is also plotted.
"""
function qqplot!(ax, x; reference_line = true)
    n = length(x)
    a = n <= 10 ? 0.375 : 0.5

    xsort = sort((x .- mean(x)) ./ std(x))
    y = quantile(Normal(), @.(((1:n) - a) / (n + 1 - 2 * a)))

    ax.scatter(y, xsort; s = 2.5, color = :black)

    if reference_line
        # Reference line
        xrange = ax.get_xlim()
        ax.plot([xrange[1], xrange[2]], [xrange[1], xrange[2]]; color = :red, linestyle = "dashed")
    end

    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
end

# Through time evolution of EM samples
@showprogress for t in 0:0.5:7
    # Find the index of the time - bit lazy
    j = findfirst(==(t), tspan)

    fig = figure()
    ax_list = fig.subplot_mosaic("""
        AAA.
        BBBC
        BBBC
        BBBC
    """)

    ax_joint = ax_list["B"]
    ax_joint.set_facecolor((0.9882352941176471, 0.9882352941176471, 0.9921568627450981))

    ax_joint.set_xlabel("°W")
    ax_joint.set_ylabel("°N")

    h, xedge, yedge, _ = ax_joint.hist2d(
        num_rels[1, j, :],
        num_rels[2, j, :];
        bins = 100,
        density = true,
        cmap = :Purples,
        rasterized = true,
    )
    ax_joint.set_xticks(ax_joint.get_xticks(), -ax_joint.get_xticks())

    ax_joint.scatter(ws[j][1], ws[j][2]; color = :red, s = 5)

    # Overlay Gaussian
    vals, evecs = eigen(Σs[j])
    θ = atand(evecs[2, 2], evecs[1, 2])
    ell_w = sqrt(vals[2])
    ell_h = sqrt(vals[1])
    for l in [1, 2, 3]
        ax_joint.add_artist(
            PyPlot.matplotlib.patches.Ellipse(;
                xy = ws[j],
                width = 2 * l * ell_w,
                height = 2 * l * ell_h,
                angle = θ,
                edgecolor = :red,
                facecolor = :none,
                linewidth = 1.0,
                linestyle = "solid",
                zorder = 2,
            ),
        )
    end

    xgrid = range(minimum(num_rels[1, j, :]); stop = maximum(num_rels[1, j, :]), length = 1000)
    ygrid = range(minimum(num_rels[2, j, :]); stop = maximum(num_rels[2, j, :]), length = 1000)

    # Pairwise longitude
    ax_lon = ax_list["A"]
    ax_lon.sharex(ax_joint)
    ax_lon.hist(
        num_rels[1, j, :];
        bins = xedge,
        align = "mid",
        density = true,
        color = (120.0 / 255.0, 115.0 / 255.0, 175.0 / 255.0),
    )
    ax_lon.plot(
        xgrid,
        pdf.(Normal(ws[j][1], sqrt(Σs[j][1, 1])), xgrid);
        color = :red,
        linewidth = 1.0,
        linestyle = "solid",
    )
    ax_lon.axis("off")

    # Pairwise latitude
    ax_lat = ax_list["C"]
    ax_lat.sharey(ax_joint)
    ax_lat.hist(
        num_rels[2, j, :];
        bins = yedge,
        align = "mid",
        density = true,
        color = (120.0 / 255.0, 115.0 / 255.0, 175.0 / 255.0),
        orientation = "horizontal",
    )
    ax_lat.plot(
        pdf.(Normal(ws[j][2], sqrt(Σs[j][2, 2])), ygrid),
        ygrid;
        color = :red,
        linewidth = 1.0,
        linestyle = "solid",
    )
    ax_lat.axis("off")

    fig.subplots_adjust(; wspace = 0, hspace = 0)
    fig.savefig("$fdir/traj_stoch_em_$t.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)

    # Sneaky normal QQ plot for diagnosis
    fig = figure()
    ax_lon = fig.add_subplot(1, 2, 1)
    ax_lat = fig.add_subplot(1, 2, 2)

    qqplot!(ax_lon, num_rels[1, j, :])
    qqplot!(ax_lat, num_rels[2, j, :])

    savefig("$fdir/traj_stoch_em_qq_$t.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)
end

# Plots of the samples over a short time span, with SSH contours for reference
begin
    end_j = findfirst(==(days[4]), tspan)
    for t in [days[1], days[2], days[3], days[4]]
        num_j = findfirst(==(t), tspan)

        fig, ax = subplots()
        ax.set_xlabel("°W")
        ax.set_ylabel("°N")

        ax.set_facecolor((0.9882352941176471, 0.9882352941176471, 0.9921568627450981))

        _, _, _, h = ax.hist2d(
            num_rels[1, num_j, :],
            num_rels[2, num_j, :];
            bins = if t > 0
                100
            else
                50
            end,
            density = true,
            cmap = :Purples,
            rasterized = true,
        )
        colorbar(h; ax = ax, location = "top", aspect = 40, label = "Density")

        # Define axis limits by range of samples at t = 3
        xlim(extrema(num_rels[1, end_j, :]))
        ylim(extrema(num_rels[2, end_j, :]))

        # SSH contours
        ax.contour(
            lon,
            lat,
            ssh_interp.(glon, glat, t)';
            colors = "grey",
            levels = 20,
            zorder = 1,
            linewidths = 0.75,
            negative_linestyles = "solid",
        )

        fig.savefig("$fdir/rels_ssh_$t.pdf"; dpi = dpi, bbox_inches = "tight")
        close(fig)
    end
end

# More diagnosis - moments through time
begin
    # Mean
    fig = figure()
    ax_lon = fig.add_subplot(2, 1, 1)
    ax_lat = fig.add_subplot(2, 1, 2)

    ax_lon.set_xlabel(L"$t$ (days)")
    ax_lon.set_ylabel(L"$\mu_\lambda$")
    ax_lat.set_xlabel(L"$t$ (days)")
    ax_lat.set_ylabel(L"$\mu_\phi$")

    ax_lon.plot(tspan, mean(num_rels[1, :, :]; dims = 2); color = :red, label = "samples")
    ax_lat.plot(tspan, mean(num_rels[2, :, :]; dims = 2); color = :red, label = "samples")

    ax_lon.plot(tspan, getindex.(ws, 1); color = :blue, label = "deterministic")
    ax_lat.plot(tspan, getindex.(ws, 2); color = :blue, label = "deterministic")

    ax_lon.legend()

    fig.savefig("$fdir/traj_stoch_em_mean.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)

    # Variances
    fig = figure()
    ax_lon = fig.add_subplot(2, 1, 1)
    ax_lat = fig.add_subplot(2, 1, 2)

    ax_lon.set_xlabel(L"$t$ (days)")
    ax_lon.set_ylabel(L"$\sigma^2_\lambda$")
    ax_lat.set_xlabel(L"$t$ (days)")
    ax_lat.set_ylabel(L"$\sigma^2_\phi$")

    ax_lon.plot(tspan, var(num_rels[1, :, :]; dims = 2); color = :red, label = "samples")
    ax_lat.plot(tspan, var(num_rels[2, :, :]; dims = 2); color = :red, label = "samples")

    ax_lon.plot(tspan, getindex.(Σs, 1, 1); color = :blue, label = "deterministic")
    ax_lat.plot(tspan, getindex.(Σs, 2, 2); color = :blue, label = "deterministic")

    ax_lon.legend()

    fig.savefig("$fdir/traj_stoch_em_var.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)
end

# Compute the Hellinger distance of the Gaussian approximation through time
Neach = 10 # Number of realisations of each Hellinger estimate
gaussians = MvNormal.(ws[2:end], Σs[2:end])
begin
    hell_dist_gauss_em = Array{Float64}(undef, length(tspan), Neach)

    Gs = Array{Float64}(undef, 2, N)

    p = Progress(Neach * length(tspan); desc = "Computing Hellinger distances...")
    for k in 1:Neach
        # Generate a new set of EM realisations
        num_sol = solve(ens, EM(), EnsembleThreads(); trajectories = N, dt = dt, saveat = tspan)
        num_rels = Array(num_sol)

        for (i, g) in enumerate(gaussians)
            # Estimate the Hellinger distance empirically - this is slow but provides a more accurate
            # and stable estimate
            rand!(g, Gs)
            hell_dist_gauss_em[i, k] = empirical_hellinger_2d(num_rels[:, i + 1, :], Gs)

            next!(p)
        end
    end
    finish!(p)
end

# Plot the Hellinger distances over time
begin
    fig, ax = subplots(; figsize = figaspect(0.5))
    ax.set_xlabel(L"$t$ (days)")
    ax.set_ylabel("Hellinger distance")

    ax.errorbar(
        tspan[2:end],
        mean(hell_dist_gauss_em; dims = 2)[1:(end - 1)];
        yerr = std(hell_dist_gauss_em; dims = 2)[1:(end - 1)] ./ sqrt(Neach),
        fmt = "k.",
        markersize = 2.5,
    )

    fig.savefig("$fdir/traj_stoch_hell_dist_bars.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)

    fig, ax = subplots(; figsize = figaspect(0.5))
    ax.set_xlabel(L"$t$ (days)")
    ax.set_ylabel("Hellinger distance")

    ax.scatter(tspan[2:end], mean(hell_dist_gauss_em; dims = 2)[1:(end - 1)]; c = "k", s = 2.5)

    fig.savefig("$fdir/traj_stoch_hell_dist.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)

    # Also report on the Hellinger distance at t = 3, for comparison to the GMMs
    println(
        "Hellinger distance at t = 3: $(mean(hell_dist_gauss_em[findfirst(==(days[4]), tspan), :]))",
    )
end

# S² along the trajectory
begin
    S2s = Array{Float64}(undef, length(tspan))
    s2s = Array{Float64}(undef, length(tspan))

    for (i, Σ) in enumerate(Σs)
        s2s[i], S2s[i] = eigvals(Σ)
    end

    fig, ax = subplots()
    ax.set_xlabel(L"$t$ (days)")
    ax.set_ylabel(L"$S^2$")

    ax.scatter(tspan, S2s; s = 2.5)
    ax.set_yscale("log")

    fig.savefig("$fdir/traj_stoch_S2.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)

    fig, ax = subplots()
    ax.set_xlabel(L"$t$ (days)")
    ax.set_ylabel(L"$s^2$")

    ax.scatter(tspan, s2s; s = 2.5)
    ax.set_yscale("log")

    fig.savefig("$fdir/traj_stoch_s2min.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)

    fig, ax = subplots()
    ax.set_xlabel(L"$t$ (days)")
    ax.set_ylabel(L"$S^2/s^2$")

    ax.scatter(tspan, S2s ./ s2s; s = 2.5)
    ax.set_yscale("log")

    fig.savefig("$fdir/traj_stoch_ratio.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)
end

# Animate the time evolution of the samples along with the Gaussian approximation. This is mainly
# for interepretation and reference, rather than trying to be a publishable figure.
begin
    using PyCall
    @pyimport matplotlib.animation as anim
    rcParams["animation.ffmpeg_path"] = "/Users/a1742080/bin/ffmpeg"

    fps = 24 * 3 / 4
    ffmt = "mp4"
    dpi = 600

    begin
        fig = figure()

        ax_hist = fig.add_subplot(2, 1, 1)
        ax_hell = fig.add_subplot(2, 1, 2)

        function makeframe(i)
            ax_hist.clear()
            ax_hist.hist2d(
                num_rels[1, i + 1, :],
                num_rels[2, i + 1, :];
                bins = 100,
                density = true,
                cmap = :Purples,
                rasterized = true,
            )

            # Overlay Gaussian
            vals, evecs = eigen(Σs[i + 1])
            θ = atand(evecs[2, 2], evecs[1, 2])
            ell_w = sqrt(vals[2])
            ell_h = sqrt(vals[1])
            for l in [1]#, 2, 3]
                ax_hist.add_artist(
                    PyPlot.matplotlib.patches.Ellipse(;
                        xy = ws[i + 1],
                        width = 2 * l * ell_w,
                        height = 2 * l * ell_h,
                        angle = θ,
                        edgecolor = :red,
                        facecolor = :none,
                        linewidth = 1.0,
                        linestyle = "dashed",
                        zorder = 2,
                    ),
                )
            end
            ax_hist.set_xlim(lon_range)
            ax_hist.set_ylim(lat_range)

            ax_hell.clear()
            ax_hell.scatter(tspan[1:(i + 1)], hell_dist_gauss_em[1:(i + 1)]; s = 2.5)
            ax_hell.set_xlim(extrema(tspan))
            ax_hell.set_ylim(extrema(hell_dist_gauss_em))
        end

        a = anim.FuncAnimation(fig, makeframe; frames = length(tspan))
        a[:save](
            "$(fdir)/traj_gauss.mp4";
            writer = anim.FFMpegWriter(;
                fps = fps,
                bitrate = -1,
                extra_args = ["-vcodec", "libx264"],
            ),
            dpi = dpi,
        )
        close(fig)
    end
end

################################## STOCHASTIC SENSITIVITY FIELDS ###################################
# Plot the stochastic sensitivity field over the plotting window, at both the resolution of the data
# and a higher resolution.
begin
    wtmp = Vector{Vector}(undef, length(tspan))
    Σtmp = Vector{Matrix}(undef, length(tspan))

    function zero_σσ!(s, _, _)
        s .= 0.0
    end

    for (str, res) in [("grid", dx), ("high", dx / 10.0)]
        x_grid = lon_range[1]:res:lon_range[2]
        y_grid = lat_range[1]:res:lat_range[2]
        inits = [[x, y] for x in x_grid, y in y_grid][:]

        S2 = Vector{Float64}(undef, length(inits))
        s2 = Vector{Float64}(undef, length(inits))
        cov2 = Vector{Float64}(undef, length(inits))
        ftle = Vector{Float64}(undef, length(inits))

        @showprogress desc = "Computing matrices for $str resolution..." for (i, x0) in
                                                                             enumerate(inits)
            gaussian_computation!(wtmp, Σtmp, 2, vel!, ∇u!, σσᵀ!, x0, zeros(2, 2), tspan)
            s2[i], S2[i] = eigvals(Σtmp[end])
            cov2[i] = Σtmp[end][1, 2]

            # Let's also compute the FTLE
            gaussian_computation!(wtmp, Σtmp, 2, vel!, ∇u!, zero_σσ!, x0, [1.0 0.0; 0.0 1.0], tspan)
            _, ftle[i] = eigvals(Σtmp[end])
        end

        # Stochastic sensitivity
        fig, ax = subplots()
        ax.set_xlabel("°W")
        ax.set_ylabel("°N")

        pc = ax.pcolormesh(
            x_grid,
            y_grid,
            reshape(S2, length(x_grid), length(y_grid))';
            cmap = :pink,
            norm = "log",
            rasterized = true,
        )
        colorbar(pc; ax = ax, location = "top", aspect = 40, label = L"S^2")

        # Overlay land
        ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

        xlim(lon_range)
        ylim(lat_range)
        ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
        fig.savefig("$fdir/S2_field_$str.pdf"; dpi = dpi, bbox_inches = "tight")
        close(fig)

        # Minimum eigenvalue
        fig, ax = subplots()
        ax.set_xlabel("°W")
        ax.set_ylabel("°N")

        pc = ax.pcolormesh(
            x_grid,
            y_grid,
            reshape(s2, length(x_grid), length(y_grid))';
            cmap = :pink,
            norm = "log",
            rasterized = true,
        )
        colorbar(pc; ax = ax, location = "top", aspect = 40, label = L"S^2")

        # Overlay land
        ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

        xlim(lon_range)
        ylim(lat_range)
        ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
        fig.savefig("$fdir/s2min_field_$str.pdf"; dpi = dpi, bbox_inches = "tight")
        close(fig)

        # Covariance
        fig, ax = subplots()
        ax.set_xlabel("°W")
        ax.set_ylabel("°N")

        pc = ax.pcolormesh(
            x_grid,
            y_grid,
            reshape(cov2, length(x_grid), length(y_grid))';
            cmap = :pink,
            norm = "log",
            rasterized = true,
        )
        colorbar(pc; ax = ax, location = "top", aspect = 40, label = L"s^2")

        # Overlay land
        ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

        xlim(lon_range)
        ylim(lat_range)
        ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
        fig.savefig("$fdir/cov_field_$str.pdf"; dpi = dpi, bbox_inches = "tight")
        close(fig)

        # Ratio between eigenvalues
        fig, ax = subplots()
        ax.set_xlabel("°W")
        ax.set_ylabel("°N")

        pc = ax.pcolormesh(
            x_grid,
            y_grid,
            reshape(s2 ./ S2, length(x_grid), length(y_grid))';
            cmap = :pink,
            norm = "log",
            rasterized = true,
        )
        colorbar(pc; ax = ax, location = "top", aspect = 40, label = L"S^2/s^2")

        # Overlay land
        ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

        xlim(lon_range)
        ylim(lat_range)
        ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
        fig.savefig("$fdir/ratio_field_$str.pdf"; dpi = dpi, bbox_inches = "tight")
        close(fig)

        # Finite-time Lyapunov exponent
        fig, ax = subplots()
        ax.set_xlabel("°W")
        ax.set_ylabel("°N")

        pc = ax.pcolormesh(
            x_grid,
            y_grid,
            reshape(ftle, length(x_grid), length(y_grid))';
            cmap = :pink,
            norm = "log",
            rasterized = true,
        )
        colorbar(pc; ax = ax, location = "top", aspect = 40, label = "FTLE stretching rate")

        # Overlay land
        ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

        xlim(lon_range)
        ylim(lat_range)
        ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
        fig.savefig("$fdir/ftle_field_$str.pdf"; dpi = dpi, bbox_inches = "tight")
        close(fig)

        # Extract some robust sets
        for R in [0.25, 0.5, 1.0, 2.0, 4.0, 6.0]
            fig, ax = subplots()
            ax.set_xlabel("°W")
            ax.set_ylabel("°N")

            ax.pcolormesh(
                x_grid,
                y_grid,
                (reshape(S2, length(x_grid), length(y_grid)) .< R)';
                cmap = "twocolor_blue",
                rasterized = true,
            )

            # Overlay land
            ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

            # Trying to get an invisible colourbar - overlay a transparent copy of the field, but
            # with a completely transparent colormap. A dirty hack, according to my Copilot overlord.
            p = ax.pcolormesh(
                x_grid,
                y_grid,
                (reshape(S2, length(x_grid), length(y_grid)) .< R)';
                cmap = ColorMap("nothing", [RGBA(0.0, 0.0, 0.0, 0.0), RGBA(0.0, 0.0, 0.0, 0.0)]),
                rasterized = true,
            )

            cb = colorbar(p; ax = ax, location = "top", aspect = 40, label = L"S^2")

            # Hide label, ticks, and outline
            cb.ax.xaxis.label.set_color(:white)
            cb.ax.tick_params(; axis = "x", colors = :white)
            cb.outline.set_visible(false)

            xlim(lon_range)
            ylim(lat_range)
            ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))

            fig.savefig("$fdir/S2_robust_$(str)_$R.pdf"; dpi = dpi, bbox_inches = "tight")
            close(fig)
        end
    end
end

###################################### MIXTURE MODEL ANALYSIS ######################################
# Grab a single-component marginal from a Gaussian mixture model
# NOTE: This will give an incorrect result if the components are not all Gaussian
gmm_marginal(mm::MixtureModel, i) =
    MixtureModel([Normal(c.μ[i], sqrt(c.Σ[i, i])) for c in mm.components], mm.prior)

# TODO: to generate error bars for Hellinger distance: First generate all the mixture models in a
#       dedicated loop. This should not take long. THEN iterate through each realisation of the
#       samples. On each iteration, we generate one set of EM samples and precalculate the kth
#       nearest distance within that set. Then we can compute the Hellinger distance for each
#       mixture model.
# This probably isn't necessary! With 10000 samples the variation is so small that these bars won't
# even be visible, at least based on the results along the trajectory. But annoyingly, the paper
# doesn't provide enough theoretical support to say this without evidence. I'll probably need to
# generate these error bar plots and include them in the appendix.

# First, a single split at different times
begin
    final_t = days[4]

    # Find the index of the time - bit lazy
    num_j = findfirst(==(final_t), tspan)

    ws = Vector{Vector{Float64}}(undef, 5)
    Σs = Vector{Matrix{Float64}}(undef, 5)
    weights = Vector{Float64}(undef, 5)

    # Just allocating the memory
    ws[1] = x₀
    Σs[1] = zeros(2, 2)

    tsplits = dt:dt:(final_t - dt)
    mixtures_split = Vector{MixtureModel}(undef, length(tsplits))
    hell_dists_split = Vector{Float64}(undef, length(tsplits))

    bwidth = dx
    Gs = Array{Float64}(undef, 2, N)
    @showprogress for (i, tsplit) in enumerate(tsplits)
        x₀ = [-60.5, 39.0]
        weights[1] = 1.0

        # Propagate initial condition to tsplit
        _solve_state_cov_forward!(ws[1], Σs[1], model, x₀, zeros(2, 2), days[1], tsplit, dt, Inf)

        # Split!
        sigma_points_symmetric!(@view(ws[2:end]), ws[1], Σs[1]; α = 0.0, include_mean = false)
        Σs = [zeros(2, 2) for _ in 1:5]
        weights[2:end] = fill(weights[1] / 5, 4)
        # Scale the original covariance and weight
        weights[1] /= 5

        # Push each component to the end
        for i in 1:5
            _solve_state_cov_forward!(
                ws[i],
                Σs[i],
                model,
                copy(ws[i]),
                copy(Σs[i]),
                tsplit,
                final_t,
                dt,
                Inf,
            )
        end

        # Construct the mixture model
        mixtures_split[i] = MixtureModel(MvNormal.(ws, Σs), weights)

        # Compute Hellinger distance
        rand!(mixtures_split[i], Gs)
        hell_dists_split[i] = empirical_hellinger_2d(num_rels[:, num_j, :], Gs)
    end

    # Plot Hellinger results
    fig, ax = subplots(; figsize = figaspect(0.5))
    ax.set_xlabel(L"$t_s$ (days)")
    ax.set_ylabel("Hellinger distance")

    ax.scatter(tsplits, hell_dists_split; s = 2.5, c = :black)

    ax.plot(
        tsplits,
        fill(0.48761, length(tsplits));
        color = :green,
        linestyle = :dashed,
        linewidth = 1.0,
    )

    fig.savefig("$fdir/hell_dist_split.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)

    best_idx = argmin(hell_dists_split)
    best_gmm = mixtures_split[best_idx]
    best_means = mean.(best_gmm.components)
    best_covs = cov.(best_gmm.components)

    println(
        "Best GMM has one split at $(dates[1] + Dates.Hour(tsplits[best_idx] * 24)) (t = $(tsplits[best_idx]) with index $best_idx) with Hellinger distance $(hell_dists_split[best_idx])",
    )

    # Plot the best mixture model
    fig = figure()
    ax_list = fig.subplot_mosaic("""
        AAA.
        BBBC
        BBBC
        BBBC
    """)

    ax_joint = ax_list["B"]
    ax_joint.set_xlabel("°W")
    ax_joint.set_ylabel("°N")

    ax_joint.set_facecolor((0.9882352941176471, 0.9882352941176471, 0.9921568627450981))

    h, xedge, yedge, _ = ax_joint.hist2d(
        num_rels[1, num_j, :],
        num_rels[2, num_j, :];
        density = true,
        cmap = :Purples,
        rasterized = true,
        bins = 100,
    )
    ax_joint.scatter(getindex.(best_means, 1), getindex.(best_means, 2); color = :red, s = 2.5)
    ax_joint.set_xticks(ax_joint.get_xticks(), -(ax_joint.get_xticks()))

    # Draw contours of the GMM
    xgrid =
        range(minimum(num_rels[1, num_j, :]); stop = maximum(num_rels[1, num_j, :]), length = 1000)
    ygrid =
        range(minimum(num_rels[2, num_j, :]); stop = maximum(num_rels[2, num_j, :]), length = 1000)
    ax_joint.contour(
        xgrid,
        ygrid,
        pdf.(Ref(best_gmm), [[x, y] for x in xgrid, y in ygrid])';
        levels = 15,
        colors = :red,
        linewidths = 0.75,
        linestyles = "solid",
    )

    ax_lon = ax_list["A"]
    ax_lon.axis("off")
    ax_lon.hist(
        num_rels[1, num_j, :];
        bins = xedge,
        align = "mid",
        density = true,
        color = (120.0 / 255.0, 115.0 / 255.0, 175.0 / 255.0),
    )
    sgrid = minimum(num_rels[1, end, :]):(1 / 10000):maximum(num_rels[1, end, :])
    ax_lon.plot(sgrid, pdf.(gmm_marginal(best_gmm, 1), sgrid); color = :red, linewidth = 1.0)
    ax_lon.set_xlim(xedge[1], xedge[end])

    ax_lat = ax_list["C"]
    ax_lat.axis("off")
    ax_lat.hist(
        num_rels[2, num_j, :];
        bins = yedge,
        align = "mid",
        density = true,
        color = (120.0 / 255.0, 115.0 / 255.0, 175.0 / 255.0),
        orientation = "horizontal",
    )
    sgrid = minimum(num_rels[2, end, :]):(1 / 10000):maximum(num_rels[2, end, :])
    ax_lat.plot(
        pdf.(gmm_marginal(best_gmm, 2), sgrid),
        sgrid; #.+ 1 / (Mm),
        color = :red,
        linewidth = 1.0,
    )
    ax_lat.set_ylim(yedge[1], yedge[end])

    fig.subplots_adjust(; wspace = 0, hspace = 0)
    fig.savefig("$fdir/gmm_split_best.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)
end

# Next - two splits! (oh my god)
begin
    final_t = days[4]

    # Find the index of the time - bit lazy
    num_j = findfirst(==(final_t), tspan)

    ws = Vector{Vector{Float64}}(undef, 25)
    Σs = Vector{Matrix{Float64}}(undef, 25)
    weights = Vector{Float64}(undef, 25)

    # Just allocating the initial memory
    ws[1] = x₀
    Σs[1] = zeros(2, 2)

    tsplits = dt:dt:(final_t - dt)
    tsplits_pairs =
        [(tsplits[i], tsplits[j]) for i in 1:length(tsplits) for j in (i + 1):length(tsplits)]
    mixtures_2split = Vector{MixtureModel}(undef, length(tsplits_pairs))
    hell_dists_2split = Vector{Float64}(undef, length(tsplits_pairs))

    bwidth = dx
    Gs = Array{Float64}(undef, 2, N)
    @showprogress for (i, (t1, t2)) in enumerate(tsplits_pairs)
        x₀ = [-60.5, 39.0]
        weights[1] = 1.0

        # Propagate initial condition to tsplit
        _solve_state_cov_forward!(ws[1], Σs[1], model, x₀, zeros(2, 2), days[1], t1, dt, Inf)

        # Split!
        sigma_points_symmetric!(@view(ws[2:5]), ws[1], Σs[1]; α = 0.0, include_mean = false)
        Σs[1:5] = [zeros(2, 2) for _ in 1:5]
        weights[2:5] = fill(weights[1] / 5, 4)
        # Scale the original covariance and weight
        weights[1] /= 5

        # Push each component to the second time and split again
        for i in 1:5
            _solve_state_cov_forward!(
                ws[i],
                Σs[i],
                model,
                copy(ws[i]),
                copy(Σs[i]),
                t1,
                t2,
                dt,
                Inf,
            )

            # Perform another split
            sigma_points_symmetric!(
                @view(ws[(2 + 4 * i):(5 + 4 * i)]),
                ws[i],
                Σs[i];
                α = 0.0,
                include_mean = false,
            )
            Σs[i] = zeros(2, 2)
            Σs[(2 + 4 * i):(5 + 4 * i)] = [zeros(2, 2) for _ in 1:4]
            weights[(2 + 4 * i):(5 + 4 * i)] = fill(weights[i] / 5, 4)
            # Scale the original weight
            weights[i] /= 5

            # Propagate the new components to the final time
            _solve_state_cov_forward!(
                ws[i],
                Σs[i],
                model,
                copy(ws[i]),
                copy(Σs[i]),
                t2,
                final_t,
                dt,
                Inf,
            )
            for j in 1:4
                _solve_state_cov_forward!(
                    ws[1 + 4 * i + j],
                    Σs[1 + 4 * i + j],
                    model,
                    copy(ws[1 + 4 * i + j]),
                    copy(Σs[1 + 4 * i + j]),
                    t2,
                    final_t,
                    dt,
                    Inf,
                )
            end
        end

        # Construct the mixture model
        mixtures_2split[i] = MixtureModel(MvNormal.(ws, Σs), weights)

        # Compute Hellinger distance
        rand!(mixtures_2split[i], Gs)
        hell_dists_2split[i] = empirical_hellinger_2d(num_rels[:, num_j, :], Gs)
    end

    # Reshape Hellinger results into a matrix - all placed in upper triangular part of matrix. Other
    # entries are NaN.
    hell_dists_2split_mat = fill(NaN, length(tsplits), length(tsplits))
    idx = 1
    for i in 1:length(tsplits)
        hell_dists_2split_mat[i, i] = hell_dists_split[i]
        for j in 1:length(tsplits)
            if i < j
                hell_dists_2split_mat[i, j] = hell_dists_2split[idx]
                # hell_dists_2split_mat[i, j] = hell_dists_2split[idx]
                idx += 1
            end
        end
    end

    best_idx = argmin(hell_dists_2split)
    best_gmm = mixtures_2split[best_idx]
    best_means = mean.(best_gmm.components)
    best_covs = cov.(best_gmm.components)

    best_t1, best_t2 = tsplits_pairs[best_idx]
    println(
        "Best GMM has two splits at $(dates[1] + Dates.Hour(best_t1 * 24)) (t = $(best_t1) with index $best_idx) and $(dates[1] + Dates.Hour(best_t2 * 24)) (t = $(best_t2)), with a Hellinger distance of $(hell_dists_2split[best_idx]).",
    )

    # Plot Hellinger results
    fig, ax = subplots(; figsize = figaspect(0.5))
    ax.set_xlabel(L"$t_{s,1}$ (days)")
    ax.set_ylabel(L"$t_{s,2}$ (days)")

    pc = ax.pcolormesh(
        [tsplits; days[4]],
        [tsplits; days[4]],
        hell_dists_2split_mat';
        cmap = :coolwarm,
        rasterized = true,
        shading = "flat",
    )
    colorbar(pc; ax = ax, location = "top", aspect = 40, label = "Hellinger distance")

    # Draw a small rectangle around the minimising box
    ax.add_artist(
        PyPlot.matplotlib.patches.Rectangle(
            (best_t1, best_t2),
            dt,
            dt;
            edgecolor = :black,
            facecolor = :none,
            linewidth = 0.75,
            zorder = 2,
        ),
    )

    fig.savefig("$fdir/hell_dist_2split.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)

    # Plot the best mixture model
    fig = figure()
    ax_list = fig.subplot_mosaic("""
    AAA.
    BBBC
    BBBC
    BBBC
    """)

    ax_joint = ax_list["B"]
    ax_joint.set_xlabel("°W")
    ax_joint.set_ylabel("°N")
    ax_joint.set_xticks(ax_joint.get_xticks(), -ax_joint.get_xticks())

    ax_joint.set_facecolor((0.9882352941176471, 0.9882352941176471, 0.9921568627450981))

    h, xedge, yedge, _ = ax_joint.hist2d(
        num_rels[1, num_j, :],
        num_rels[2, num_j, :];
        density = true,
        cmap = :Purples,
        rasterized = true,
        bins = 100,
    )
    ax_joint.scatter(getindex.(best_means, 1), getindex.(best_means, 2); color = :red, s = 2.5)

    # Draw contours of the GMM
    xgrid =
        range(minimum(num_rels[1, num_j, :]); stop = maximum(num_rels[1, num_j, :]), length = 1000)
    ygrid =
        range(minimum(num_rels[2, num_j, :]); stop = maximum(num_rels[2, num_j, :]), length = 1000)
    ax_joint.contour(
        xgrid,
        ygrid,
        pdf.(Ref(best_gmm), [[x, y] for x in xgrid, y in ygrid])';
        levels = 15,
        colors = :red,
        linewidths = 0.75,
        linestyles = "solid",
    )

    ax_lon = ax_list["A"]
    ax_lon.axis("off")
    ax_lon.hist(
        num_rels[1, num_j, :];
        bins = xedge,
        align = "mid",
        density = true,
        color = (120.0 / 255.0, 115.0 / 255.0, 175.0 / 255.0),
    )
    sgrid = minimum(num_rels[1, end, :]):(1 / 10000):maximum(num_rels[1, end, :])
    ax_lon.plot(sgrid, pdf.(gmm_marginal(best_gmm, 1), sgrid); color = :red, linewidth = 1.0)
    ax_lon.set_xlim(xedge[1], xedge[end])

    ax_lat = ax_list["C"]
    ax_lat.axis("off")
    ax_lat.hist(
        num_rels[2, num_j, :];
        bins = yedge,
        align = "mid",
        density = true,
        color = (120.0 / 255.0, 115.0 / 255.0, 175.0 / 255.0),
        orientation = "horizontal",
    )
    sgrid = minimum(num_rels[2, end, :]):(1 / 10000):maximum(num_rels[2, end, :])
    ax_lat.plot(pdf.(gmm_marginal(best_gmm, 2), sgrid), sgrid; color = :red, linewidth = 1.0)
    ax_lat.set_ylim(yedge[1], yedge[end])

    fig.subplots_adjust(; wspace = 0, hspace = 0)
    fig.savefig("$fdir/gmm_2split_best.pdf"; bbox_inches = "tight", dpi = dpi)
    close(fig)
end

