# The original 2D containers were designed for CPU arrays specifically. Here, 
# we create the corresponding 2D containers for holding GPU arrays, which helps 
# avoid multiple data transfers between the CPU and GPU.

# Container data structure for DG elements
mutable struct ElementContainerGPU2D{RealT <: Real, uEltype <: Real} <: AbstractContainer
    inverse_jacobian::CuArray{RealT, 1}
    node_coordinates::CuArray{RealT, 4}
    surface_flux_values::CuArray{uEltype, 4}
    cell_ids::CuArray{Int, 1}

    # Inner constructor
    function ElementContainerGPU2D{RealT, uEltype}(dims_inverse_jacobian::NTuple{1, Int},
                                                   dims_node_coordinates::NTuple{4, Int},
                                                   dims_surface_flux_values::NTuple{4, Int},
                                                   dims_cell_ids::NTuple{1, Int}) where {
                                                                                         RealT,
                                                                                         uEltype <:
                                                                                         Real
                                                                                         }
        inverse_jacobian = CuArray{RealT}(undef, dims_inverse_jacobian...)
        node_coordinates = CuArray{RealT}(undef, dims_node_coordinates...)
        surface_flux_values = CuArray{uEltype}(undef, dims_surface_flux_values...)
        cell_ids = CuArray{Int}(undef, dims_cell_ids...)

        new(inverse_jacobian, node_coordinates, surface_flux_values, cell_ids)
    end
end

Base.eltype(elements::ElementContainerGPU2D) = eltype(elements.surface_flux_values)
@inline nelements(elements::ElementContainerGPU2D) = length(elements.cell_ids)

# Copy arrays from DG elements to GPU
function copy_elements(elements::ElementContainer2D)
    dims_inverse_jacobian = size(elements.inverse_jacobian)
    dims_node_coordinates = size(elements.node_coordinates)
    dims_surface_flux_values = size(elements.surface_flux_values)
    dims_cell_ids = size(elements.cell_ids)

    RealT = eltype(elements.inverse_jacobian)
    uEltype = eltype(elements.surface_flux_values)

    elements_gpu = ElementContainerGPU2D{RealT, uEltype}(dims_inverse_jacobian,
                                                         dims_node_coordinates,
                                                         dims_surface_flux_values,
                                                         dims_cell_ids)

    elements_gpu.inverse_jacobian = CuArray(elements.inverse_jacobian)
    elements_gpu.node_coordinates = CuArray(elements.node_coordinates)
    elements_gpu.surface_flux_values = CuArray(elements.surface_flux_values)
    elements_gpu.cell_ids = CuArray(elements.cell_ids)

    return elements_gpu
end

# Container data structure for DG interfaces
mutable struct InterfaceContainerGPU2D{uEltype <: Real} <: AbstractContainer
    u::CuArray{uEltype, 4}
    neighbor_ids::CuArray{Int, 2}
    orientations::CuArray{Int, 1}

    # Inner constructor
    function InterfaceContainerGPU2D{uEltype}(dims_u::NTuple{4, Int},
                                              dims_neighbor_ids::NTuple{2, Int},
                                              dims_orientations::NTuple{1, Int}) where {
                                                                                        uEltype <:
                                                                                        Real
                                                                                        }
        u = CuArray{uEltype}(undef, dims_u...)
        neighbor_ids = CuArray{Int}(undef, dims_neighbor_ids...)
        orientations = CuArray{Int}(undef, dims_orientations...)

        new(u, neighbor_ids, orientations)
    end
end

@inline ninterfaces(interfaces::InterfaceContainerGPU2D) = length(interfaces.orientations)

# Copy arrays from DG interfaces to GPU
function copy_interfaces(interfaces::InterfaceContainer2D)
    dims_u = size(interfaces.u)
    dims_neighbor_ids = size(interfaces.neighbor_ids)
    dims_orientations = size(interfaces.orientations)

    uEltype = eltype(interfaces.u)

    interfaces_gpu = InterfaceContainerGPU2D{uEltype}(dims_u,
                                                      dims_neighbor_ids,
                                                      dims_orientations)

    interfaces_gpu.u = CuArray(interfaces.u)
    interfaces_gpu.neighbor_ids = CuArray(interfaces.neighbor_ids)
    interfaces_gpu.orientations = CuArray(interfaces.orientations)

    return interfaces_gpu
end

# Container data structure for DG boundaries
mutable struct BoundaryContainerGPU2D{RealT <: Real, uEltype <: Real} <: AbstractContainer
    u::CuArray{uEltype, 4}
    neighbor_ids::CuArray{Int, 1}
    orientations::CuArray{Int, 1}
    neighbor_sides::CuArray{Int, 1}
    node_coordinates::CuArray{RealT, 3}
    n_boundaries_per_direction::CuArray{Int, 1}

    # Inner constructor
    function BoundaryContainerGPU2D{RealT, uEltype}(dims_u::NTuple{4, Int},
                                                    dims_neighbor_ids::NTuple{1, Int},
                                                    dims_orientations::NTuple{1, Int},
                                                    dims_neighbor_sides::NTuple{1, Int},
                                                    dims_node_coordinates::NTuple{3, Int},
                                                    dims_n_boundaries_per_direction::NTuple{1, Int}) where {
                                                                                                            RealT,
                                                                                                            uEltype <:
                                                                                                            Real
                                                                                                            }
        u = CuArray{uEltype}(undef, dims_u...)
        neighbor_ids = CuArray{Int}(undef, dims_neighbor_ids...)
        orientations = CuArray{Int}(undef, dims_orientations...)
        neighbor_sides = CuArray{Int}(undef, dims_neighbor_sides...)
        node_coordinates = CuArray{RealT}(undef, dims_node_coordinates...)
        n_boundaries_per_direction = CuArray{Int}(undef, dims_n_boundaries_per_direction...)

        new(u, neighbor_ids, orientations, neighbor_sides, node_coordinates,
            n_boundaries_per_direction)
    end
end

# Copy arrays from DG boundaries to GPU
function copy_boundaries(boundaries::BoundaryContainer2D)
    dims_u = size(boundaries.u)
    dims_neighbor_ids = size(boundaries.neighbor_ids)
    dims_orientations = size(boundaries.orientations)
    dims_neighbor_sides = size(boundaries.neighbor_sides)
    dims_node_coordinates = size(boundaries.node_coordinates)
    dims_n_boundaries_per_direction = size(boundaries.n_boundaries_per_direction)

    RealT = eltype(boundaries.node_coordinates)
    uEltype = eltype(boundaries.u)

    boundaries_gpu = BoundaryContainerGPU2D{RealT, uEltype}(dims_u, dims_neighbor_ids,
                                                            dims_orientations, dims_neighbor_sides,
                                                            dims_node_coordinates,
                                                            dims_n_boundaries_per_direction)

    boundaries_gpu.u = CuArray(boundaries.u)
    boundaries_gpu.neighbor_ids = CuArray(boundaries.neighbor_ids)
    boundaries_gpu.orientations = CuArray(boundaries.orientations)
    boundaries_gpu.neighbor_sides = CuArray(boundaries.neighbor_sides)
    boundaries_gpu.node_coordinates = CuArray(boundaries.node_coordinates)
    boundaries_gpu.n_boundaries_per_direction = CuArray(boundaries.n_boundaries_per_direction)

    return boundaries_gpu
end

# Container data structure for DG mortars
mutable struct L2MortarContainerGPU2D{uEltype <: Real} <: AbstractContainer
    u_upper::CuArray{uEltype, 4}
    u_lower::CuArray{uEltype, 4}
    neighbor_ids::CuArray{Int, 2}
    large_sides::CuArray{Int, 1} # Large sides: left -> 1, right -> 2
    orientations::CuArray{Int, 1}

    # Inner constructor
    function L2MortarContainerGPU2D{uEltype}(dims_u_upper::NTuple{4, Int},
                                             dims_u_lower::NTuple{4, Int},
                                             dims_neighbor_ids::NTuple{2, Int},
                                             dims_large_sides::NTuple{1, Int},
                                             dims_orientations::NTuple{1, Int}) where {
                                                                                       uEltype <:
                                                                                       Real
                                                                                       }
        u_upper = CUDA.zeros(dims_u_upper...) # initialize with zeros
        u_lower = CUDA.zeros(dims_u_lower...) # initialize with zeros
        neighbor_ids = CuArray{Int}(undef, dims_neighbor_ids...)
        large_sides = CuArray{Int}(undef, dims_large_sides...)
        orientations = CuArray{Int}(undef, dims_orientations...)

        new(u_upper, u_lower, neighbor_ids, large_sides, orientations)
    end
end

@inline nmortars(l2mortars::L2MortarContainerGPU2D) = length(l2mortars.orientations)

# Copy arrays from DG mortars to GPU
function copy_mortars(mortars::L2MortarContainer2D)
    dims_u_upper = size(mortars.u_upper)
    dims_u_lower = size(mortars.u_lower)
    dims_neighbor_ids = size(mortars.neighbor_ids)
    dims_large_sides = size(mortars.large_sides)
    dims_orientations = size(mortars.orientations)

    uEltype = eltype(mortars.u_upper)

    mortars_gpu = L2MortarContainerGPU2D{uEltype}(dims_u_upper, dims_u_lower,
                                                  dims_neighbor_ids, dims_large_sides,
                                                  dims_orientations)

    mortars_gpu.neighbor_ids = CuArray(mortars.neighbor_ids)
    mortars_gpu.large_sides = CuArray(mortars.large_sides)
    mortars_gpu.orientations = CuArray(mortars.orientations)

    return mortars_gpu
end
