# Some common functions that are shared between the solvers.

# Copy data from CPU to GPU
function reset_du!(du::CuArray)
    du_zero = zero(du)
    du .= du_zero # no scalar indexing

    return nothing
end

# Set diagonal entries of a matrix to zeros
function set_diagonal_to_zero!(A::Array{Float64})
    n = min(size(A)...)
    for i in 1:n
        A[i, i] = 0.0 # change back to `Float32`
    end
    return nothing
end

# Kernel for getting last and first indices
function last_first_indices_kernel!(lasts, firsts, n_boundaries_per_direction)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (i <= length(n_boundaries_per_direction))
        @inbounds begin
            for ii in 1:i
                lasts[i] += n_boundaries_per_direction[ii]
            end
            firsts[i] = lasts[i] - n_boundaries_per_direction[i] + 1
        end
    end

    return nothing
end

# Kernel for counting elements for DG-only and blended DG-FV volume integral
function pure_blended_element_count_kernel!(element_ids_dg, element_ids_dgfv, alpha, atol)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (i <= length(alpha))
        dg_only = isapprox(alpha[i], 0, atol = atol)

        if dg_only # bad
            element_ids_dg[i] = i
        else
            element_ids_dgfv[i] = i
        end
    end

    return nothing
end
