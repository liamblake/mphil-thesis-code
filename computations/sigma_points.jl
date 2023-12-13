using LinearAlgebra

"""
    sigma_points_symmetric!(dest::AbstractVector, μ::AbstractVector, Σ::AbstractArray)

Compute a symmetric set of 2n+1 sigma points, where n is the dimension, given a mean μ and
covariance Σ. The points are such that the mean is μ and the covariance is Σ. The sigma points are
stored in dest, which will be a vector of vectors.
"""
function sigma_points_symmetric!(
    dest::AbstractVector,
    μ::AbstractVector,
    Σ::AbstractArray;
    α::Real = 0.0,
    include_mean::Bool = true,
)
    n = length(μ)
    if size(Σ) != (n, n)
        throw(DimensionMismatch("Covariance Σ must have dimension ($n, $n), but got $(size(Σ))."))
    end

    if !(0 ≤ α ≤ 1)
        throw(ArgumentError("Scaling α must be between 0 and 1, got $α"))
    end

    if include_mean && size(dest) != (2 * n + 1,)
        throw(DimensionMismatch("dest must have dimension ($(2*n+1),), but not $(size(dest))."))
    elseif !include_mean && size(dest) != (2 * n,)
        throw(DimensionMismatch("dest must have dimension ($(2*n),), but not $(size(dest))."))
    end

    # Matrix square root - use eigendecomposition - P² = Σ
    P = sqrt(Symmetric(Σ))

    scaling = sqrt((2 * n + 1) * (1 - α) / 2)
    for i in 1:n
        dest[i] = μ + scaling * P[:, i]
        dest[n + i] = μ - scaling * P[:, i]
    end

    if include_mean
        dest[end] = μ
    end

    nothing
end

