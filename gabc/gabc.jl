"""
Compute stochastic sensitivity for the Gromeka-Arnold-Beltrami-Childress flow with identity
diffusion:
    dxₜ = (Asin(zₜ) + Ccos(yₜ))dt + dWₜ¹
    dyₜ = (Bsin(xₜ) + Acos(zₜ))dt + dWₜ²
    dzₜ = (Csin(yₜ) + Bcos(xₜ))dt + dWₜ³
where Wₜ¹, Wₜ², Wₜ³ are independent Brownian motions.
"""

using PyPlot
using ProgressMeter

include("../pyplot_setup.jl")
include("../computations/gaussian_computation.jl")

# Parameter values
A = sqrt(3)
B = sqrt(2)
C = 1

# Output
fdir = "../../thesis/chp04_paper_numerics/figures/gabc"

# First standalone plot: streamlines for integrable case
begin
    # Streamfunction
    ψ = (x, z) -> -sqrt(3) * cos(z) - sqrt(2) * sin(x)

    # Evaluate
    xgrid = range(0; stop = 2π, length = 1000)
    zgrid = range(0; stop = 2π, length = 1000)
    ψgrid = [ψ(x, z) for x in xgrid, z in zgrid]

    fig, ax = subplots(; figsize = [6.4, 4 / 5 * 4.8])
    ax.set_xlabel(L"y_1")
    ax.set_ylabel(L"y_3")

    ax.contour(
        xgrid,
        zgrid,
        ψgrid';
        levels = 10,
        colors = "black",
        linewidths = 0.5,
        negative_linestyles = "solid",
    )

    # Manually plot the heteroclinic orbits
    for c in [sqrt(3) - sqrt(2), sqrt(2) - sqrt(3)]
        ax.plot(
            xgrid,
            @.(acos(-(c + sqrt(2) * sin(xgrid)) / sqrt(3)));
            color = "black",
            linewidth = 0.5,
        )
        ax.plot(
            xgrid,
            @.(2π - acos(-(c + sqrt(2) * sin(xgrid)) / sqrt(3)));
            color = "black",
            linewidth = 0.5,
        )
    end

    # Stagnation-like points
    ax.scatter(
        [π / 2, π / 2, π / 2, 3π / 2, 3π / 2, 3π / 2],
        [0, π, 2π, 0, π, 2π];
        color = "black",
        s = 7.5,
    )

    fig.savefig("$fdir/streamlines_integrable.pdf")
    close(fig)
end

# Define the ABC flow
function u!(s, x, _)
    s[1] = A * sin(x[3]) + C * cos(x[2])
    s[2] = B * sin(x[1]) + A * cos(x[3])
    s[3] = C * sin(x[2]) + B * cos(x[1])
end

function ∇u!(s, x, _)
    s[1, 1] = 0.0
    s[1, 2] = -C * sin(x[2])
    s[1, 3] = A * cos(x[3])

    s[2, 1] = B * cos(x[1])
    s[2, 2] = 0.0
    s[2, 3] = -A * sin(x[3])

    s[3, 1] = -C * cos(x[2])
    s[3, 2] = B * sin(x[1])
    s[3, 3] = 0.0
end

function σ!(s, _, _)
    s .= 0.0
    s[1, 1] = 1.0
    s[2, 2] = 1.0
    s[3, 3] = 1.0
end

begin
    # Number of points in each dimension
    npoints = 201
    zs = range(0; stop = 2π, length = npoints)
    xs = range(0; stop = 2π, length = npoints)
    ys = range(0; stop = 2π, length = npoints)

    X = repeat(xs, 1, length(ys), length(zs))
    Y = repeat(ys', length(xs), 1, length(zs))
    Z = repeat(reshape(zs, 1, 1, length(zs)), length(xs), length(ys), 1)

    dt = 0.01
    T = 3.0

    S2s = fill(NaN, length(xs), length(ys), length(zs))
    w_tmp = Vector{Float64}(undef, 3)
    Σ_tmp = Matrix{Float64}(undef, 3, 3)

    ts = range(0; stop = T, step = dt)

    # Only compute the value on each of the slices, in a similar fashion to Figures 6-13 of Dombre
    # et al. (1986).
    slices = [0, π / 4, π / 2, 3π / 4, π, 5π / 4, 3π / 2, 7π / 4]
    slices_idx = [findfirst(isequal(s), zs) for s in slices]
    @assert slices == zs[slices_idx]

    p = Progress(length(xs) * length(ys) * length(zs))
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(ys)
            for (k, z) in enumerate(zs)
                # Only compute if the point is on a slice
                if (x in slices) || (y in slices) || (z in slices)
                    _solve_state_cov_forward!(
                        w_tmp,
                        Σ_tmp,
                        3,
                        u!,
                        ∇u!,
                        σ!,
                        [x, y, z],
                        zeros(3, 3),
                        0.0,
                        T,
                        dt,
                        Inf,
                    )
                    # gaussian_computation!(w_tmp, Σ_tmp, 3, u!, ∇u!, σ!, [x, y, z], zeros(3, 3), ts)
                    S2s[i, j, k] = opnorm(Σ_tmp)#[end])
                end
                next!(p)
            end
        end
    end
    finish!(p)

    # # Visualise three faces of the cube
    # begin
    #     fig = figure()
    #     ax = fig.add_subplot(111; projection = "3d")

    #     # Setup S2 as a colormap
    #     minn, maxx = extrema(S2s[.!isnan.(S2s)])
    #     fnorm = PyPlot.matplotlib.colors.LogNorm(minn, maxx)
    #     m = PyPlot.cm.ScalarMappable(; norm = fnorm, cmap = "pink")
    #     m.set_array([])

    #     # Plot each face
    #     ax.plot_surface(
    #         X[:, :, end],
    #         Y[:, :, end],
    #         Z[:, :, end];
    #         facecolors = m.to_rgba(S2s[:, :, end]),
    #         shade = false,
    #         rcount = npoints,
    #         ccount = npoints,
    #         zorder = 0,
    #     )
    #     ax.plot_surface(
    #         X[:, 1, :],
    #         Y[:, 1, :],
    #         Z[:, 1, :];
    #         facecolors = m.to_rgba(S2s[:, 1, :]),
    #         shade = false,
    #         rcount = npoints,
    #         ccount = npoints,
    #         zorder = 0,
    #     )
    #     ax.plot_surface(
    #         X[end, :, :],
    #         Y[end, :, :],
    #         Z[end, :, :];
    #         facecolors = m.to_rgba(S2s[end, :, :]),
    #         shade = false,
    #         rcount = npoints,
    #         ccount = npoints,
    #         zorder = 0,
    #     )

    #     xmin, xmax = extrema(xs)
    #     ymin, ymax = extrema(ys)
    #     zmin, zmax = extrema(zs)
    #     ax.set(; xlim = [xmin, xmax], ylim = [ymin, ymax], zlim = [zmin, zmax])

    #     ax.plot([xmax, xmax], [ymin, ymax], zmax; color = "grey", linewidth = 1.5, zorder = 3)
    #     ax.plot([xmin, xmax], [ymin, ymin], zmax; color = "grey", linewidth = 1.5, zorder = 3)
    #     ax.plot(
    #         [xmax, xmax],
    #         [ymin, ymin],
    #         [zmin, zmax];
    #         color = "grey",
    #         linewidth = 1.5,
    #         zorder = 3,
    #     )

    #     ax.set_xlabel(L"y_1"; labelpad = -4)
    #     ax.set_ylabel(L"y_2"; labelpad = -4)
    #     ax.set_zlabel(L"y_3"; labelpad = -4)

    #     ax.grid(false)
    #     ax.margins(0)

    #     ax.tick_params(; axis = "y", pad = -2)
    #     ax.tick_params(; axis = "z", pad = -2)
    #     ax.tick_params(; axis = "x", pad = -2)

    #     # Set zoom and angle view
    #     ax.view_init(30, -45, 0)
    #     ax.set_box_aspect((1.0, 1.0, 1.0); zoom = 1.0)

    #     fig.subplots_adjust()
    #     fig.colorbar(
    #         m;
    #         ax = ax,
    #         location = "top",
    #         fraction = 0.05,
    #         pad = 0.0,
    #         aspect = 40,
    #         shrink = 0.6,
    #         label = L"S^2",
    #     )

    #     fig.savefig("$fdir/S2_box_$(A)_$(B)_$(C).pdf")
    #     close(fig)

    #     # Crop the figure with the pdfcrop tool
    #     run(
    #         `pdfcrop --margins '0 0 0 0' $fdir/S2_box_$(A)_$(B)_$(C).pdf $fdir/S2_box_$(A)_$(B)_$(C)_cropped.pdf`,
    #     )
    # end
end

# Plot each set of slices
begin
    # Setup a colourmap for EVERY plot
    minn, maxx = extrema(S2s[slices_idx, :, :])
    fnorm = PyPlot.matplotlib.colors.LogNorm(minn, maxx)

    # Change up the order
    slices = [3π / 4, 7π / 4, π / 2, 3π / 2, π / 4, 5π / 4, 0, π] #[0, π / 4, π / 2, 3π / 4, π, 5π / 4, 3π / 2, 7π / 4]
    slices_idx = [findfirst(isequal(s), zs) for s in slices]

    # YZ
    fig = figure(; figsize = figaspect(2))

    # Simulataneously produce a key
    fig_key = figure(; figsize = figaspect(2))

    # Common axis labels: https://stackoverflow.com/a/6981055
    # Create a big invisible axis
    ax_labels = fig_key.add_subplot(111; frameon = false)
    ax_labels.set_xlabel(L"y_2")
    ax_labels.set_ylabel(L"y_3")
    ax_labels.set_xticks([])
    ax_labels.set_yticks([])

    pc = 1

    ls = [
        "\\frac{3\\pi}{4}",
        "\\frac{7\\pi}{4}",
        "\\frac{\\pi}{2}",
        "\\frac{3\\pi}{2}",
        "\\frac{\\pi}{4}",
        "\\frac{5\\pi}{4}",
        "0",
        "\\pi",
    ]
    for (i, s) in enumerate(slices_idx)
        ax = fig.add_subplot(4, 2, i)
        axk = fig_key.add_subplot(4, 2, i)

        # Key
        axk.text(0.5, 0.5, L"y_1 = %$(ls[i])"; ha = "center", va = "center")
        axk.set_xticks([])
        axk.set_yticks([])

        if i == 1
            ax.set_xlabel(L"y_2")
            ax.set_ylabel(L"y_3")

            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xticks([0, π / 2, π, 3π / 2, 2π])
            ax.set_xticklabels(["0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"])

            ax.set_yticks([0, π / 2, π, 3π / 2, 2π])
            ax.set_yticklabels(["0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"])
        else
            # Remove ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
        end

        ax.pcolormesh(
            Z[s, :, :],
            Y[s, :, :],
            S2s[s, :, :]';
            shading = "auto",
            cmap = "pink",
            norm = fnorm,
            rasterized = true,
        )
    end

    # Decrease spacing between plots
    fig.subplots_adjust(; hspace = 0.05, wspace = 0.05)

    fig.savefig("$fdir/S2_slices_yz.pdf"; bbox_inches = "tight", dpi = 600)
    close(fig)

    fig_key.subplots_adjust(; hspace = 0.05, wspace = 0.05)
    fig_key.savefig("$fdir/S2_slices_yz_key.pdf"; bbox_inches = "tight", dpi = 600)
    close(fig_key)

    # XY
    fig = figure(; figsize = figaspect(2))

    # Simulataneously produce a key
    fig_key = figure(; figsize = figaspect(2))

    # Common axis labels: https://stackoverflow.com/a/6981055
    # Create a big invisible axis
    ax_labels = fig_key.add_subplot(111; frameon = false)
    ax_labels.set_xlabel(L"y_1")
    ax_labels.set_ylabel(L"y_2")
    ax_labels.set_xticks([])
    ax_labels.set_yticks([])

    for (i, s) in enumerate(slices_idx)
        ax = fig.add_subplot(4, 2, i)

        # Key
        axk = fig_key.add_subplot(4, 2, i)
        axk.text(0.5, 0.5, L"y_3 = %$(ls[i])"; ha = "center", va = "center")
        axk.set_xticks([])
        axk.set_yticks([])

        if i == 1
            ax.set_xlabel(L"y_1")
            ax.set_ylabel(L"y_2")

            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

            ax.set_xticks([0, π / 2, π, 3π / 2, 2π])
            ax.set_xticklabels(["0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"])

            ax.set_yticks([0, π / 2, π, 3π / 2, 2π])
            ax.set_yticklabels(["0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"])
        else
            # Remove ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
        end

        ax.pcolormesh(
            Y[:, :, s],
            X[:, :, s],
            S2s[:, :, s]';
            shading = "auto",
            cmap = "pink",
            norm = "log",
            rasterized = true,
        )
    end

    # Decrease spacing between plots
    fig.subplots_adjust(; hspace = 0.05, wspace = 0.05)

    fig.savefig("$fdir/S2_slices_xy.pdf"; bbox_inches = "tight", dpi = 600)
    close(fig)

    fig_key.subplots_adjust(; hspace = 0.05, wspace = 0.05)
    fig_key.savefig("$fdir/S2_slices_xy_key.pdf"; bbox_inches = "tight", dpi = 600)
    close(fig_key)

    # XZ
    fig = figure(; figsize = figaspect(2))

    # Simulataneously produce a key
    fig_key = figure(; figsize = figaspect(2))

    # Common axis labels: https://stackoverflow.com/a/6981055
    # Create a big invisible axis
    ax_labels = fig_key.add_subplot(111; frameon = false)
    ax_labels.set_xlabel(L"y_1")
    ax_labels.set_ylabel(L"y_3")
    ax_labels.set_xticks([])
    ax_labels.set_yticks([])

    # Steal the last colourmap
    pc = 1

    for (i, s) in enumerate(slices_idx)
        ax = fig.add_subplot(4, 2, i)

        # Key
        axk = fig_key.add_subplot(4, 2, i)
        axk.text(0.5, 0.5, L"y_2 = %$(ls[i])"; ha = "center", va = "center")
        axk.set_xticks([])
        axk.set_yticks([])

        if i == 1
            ax.set_xlabel(L"y_1")
            ax.set_ylabel(L"y_3")

            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

            ax.set_xticks([0, π / 2, π, 3π / 2, 2π])
            ax.set_xticklabels(["0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"])

            ax.set_yticks([0, π / 2, π, 3π / 2, 2π])
            ax.set_yticklabels(["0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"])
        else
            # Remove ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
        end

        pc = ax.pcolormesh(
            Z[:, s, :],
            X[:, s, :],
            S2s[:, s, :]';
            shading = "auto",
            cmap = "pink",
            norm = "log",
            rasterized = true,
        )
    end

    # Decrease spacing between plots
    fig.subplots_adjust(; hspace = 0.05, wspace = 0.05)

    fig.savefig("$fdir/S2_slices_xz.pdf"; bbox_inches = "tight", dpi = 600)
    close(fig)

    fig_key.subplots_adjust(; hspace = 0.05, wspace = 0.05)
    fig_key.savefig("$fdir/S2_slices_xz_key.pdf"; bbox_inches = "tight", dpi = 600)
    close(fig_key)

    # Create a separate PDF with the colourbar
    fig = figure(; figsize = figaspect(2))
    fig.gca().set_visible(false)
    cb = fig.colorbar(
        PyPlot.cm.ScalarMappable(; norm = fnorm, cmap = "pink");
        orientation = "vertical",
        location = "right",
        # label = L"S^2",
        aspect = 40,
    )
    cb.set_label(L"S^2"; rotation = 0)

    fig.savefig("$fdir/S2_slices_colourbar.pdf"; bbox_inches = "tight", dpi = 600)
    close(fig)
end
