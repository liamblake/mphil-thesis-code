using Distributions
using NearestNeighbors
using SpecialFunctions

"""
    empirical_hellinger_2d(x, y; k = 5)

Estimate the Hellinger distance between two distributions `x` and `y` using the empirical estimator
by Ding and Mullhaupt. Each column of `x` and `y` corresponds to a two-dimensional sample of the
distributions in question. The number of nearest neighbours to use in the empirical PDF estimate is
specified by `k`.
"""
function empirical_hellinger_2d(x, y; k = 5)
    N = size(x, 2)

    tree_x = BruteTree(x, Euclidean())
    tree_y = BruteTree(y, Euclidean())

    # note k + 1, since the point itself will always be the closest neighbour
    _, dists_xx = knn(tree_x, x, k + 1, true)
    _, dists_xy = knn(tree_y, x, k, true)
    _, dists_yx = knn(tree_x, y, k, true)
    _, dists_yy = knn(tree_y, y, k + 1, true)

    kdist_xx = getindex.(dists_xx, k + 1)
    kdist_xy = getindex.(dists_xy, k)
    kdist_yx = getindex.(dists_yx, k)
    kdist_yy = getindex.(dists_yy, k + 1)

    H²1 =
        1 -
        sqrt(N) * factorial(k - 1)^2 / (N^(3 / 2) * gamma(k - 0.5) * gamma(k + 0.5)) *
        sum(kdist_xx ./ kdist_xy)
    H²2 =
        1 -
        sqrt(N) * factorial(k - 1)^2 / (N^(3 / 2) * gamma(k - 0.5) * gamma(k + 0.5)) *
        sum(kdist_yy ./ kdist_yx)

    return sqrt((abs(H²1) + abs(H²2)) / 2)
end

############################################# Testing ##############################################
# function test_hellinger_2d()
#     m1 = [0.0, 1.2]
#     S1 = I
#     d1 = MvNormal(m1, S1)

#     m2 = [1.0, -0.2]
#     S2 = [5.0 -1.2; -1.2 3.2]
#     d2 = MvNormal(m2, S2)

#     N = 10000
#     hd = empirical_hellinger_2d(rand(d1, N), rand(d2, N))

#     println("Estimated Hellinger distance: $hd")
#     println(
#         "True Hellinger distance: $(sqrt(1 - det(S1)^(1/4) * det(S2)^(1/4) / det((S1 + S2)/2)^(1/2) * exp(-1 / 8 * (m1 - m2)' * inv((S1 + S2)/2) * (m1 - m2))))",
#     )
# end

# using ProfileView, BenchmarkTools
# @benchmark test_hellinger_2d()
# VSCodeServer.@profview test_hellinger_2d()

