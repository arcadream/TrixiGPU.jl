module BurgersRarefaction1D

using Trixi, TrixiCUDA
using Test

include("../test_macros.jl")

@testset "Burgers Rarefaction 1D" begin
    equations = InviscidBurgersEquation1D()

    basis = LobattoLegendreBasis(3)
    # Use shock capturing techniques to suppress oscillations at discontinuities
    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max = 1.0,
                                             alpha_min = 0.001,
                                             alpha_smooth = true,
                                             variable = first)

    volume_flux = flux_ec
    surface_flux = flux_lax_friedrichs

    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg = volume_flux,
                                                     volume_flux_fv = surface_flux)

    solver = DGSEM(basis, surface_flux, volume_integral)

    coordinate_min = 0.0
    coordinate_max = 1.0

    mesh = TreeMesh(coordinate_min, coordinate_max,
                    initial_refinement_level = 6,
                    n_cells_max = 10_000,
                    periodicity = false)

    # Discontinuous initial condition (Riemann Problem) leading to a rarefaction fan.
    function initial_condition_rarefaction(x, t, equation::InviscidBurgersEquation1D)
        scalar = x[1] < 0.5 ? 0.5 : 1.5

        return SVector(scalar)
    end

    boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_rarefaction)

    function boundary_condition_outflow(u_inner, orientation, normal_direction, x, t,
                                        surface_flux_function,
                                        equations::InviscidBurgersEquation1D)
        # Calculate the boundary flux entirely from the internal solution state
        flux = Trixi.flux(u_inner, normal_direction, equations)

        return flux
    end

    boundary_conditions = (x_neg = boundary_condition_inflow,
                           x_pos = boundary_condition_outflow)

    initial_condition = initial_condition_rarefaction

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions = boundary_conditions)
    semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver,
                                               boundary_conditions = boundary_conditions)

    tspan = (0.0, 0.2)
    t = t_gpu = 0.0

    # Semi on CPU
    (; mesh, equations, boundary_conditions, source_terms, solver, cache) = semi

    # Semi on GPU
    equations_gpu = semi_gpu.equations
    mesh_gpu, solver_gpu, cache_gpu = semi_gpu.mesh, semi_gpu.solver, semi_gpu.cache
    boundary_conditions_gpu = semi_gpu.boundary_conditions
    source_terms_gpu = semi_gpu.source_terms

    # ODE on CPU
    ode = semidiscretize(semi, tspan)
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
    du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

    # ODE on GPU
    ode_gpu = semidiscretizeGPU(semi_gpu, tspan)
    u_gpu = copy(ode_gpu.u0)
    du_gpu = similar(u_gpu)

    # Tests for components initialization
    @test_approx (u_gpu, u)
    # du is initlaizaed as undefined, cannot test now

    # Tests for semidiscretization process
    TrixiCUDA.reset_du!(du_gpu)
    Trixi.reset_du!(du, solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                    Trixi.have_nonconservative_terms(equations_gpu),
                                    equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                    cache_gpu)
    Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                equations, solver.volume_integral, solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
    Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
    @test_approx (cache_gpu.interfaces.u, cache.interfaces.u)

    TrixiCUDA.cuda_interface_flux!(mesh_gpu,
                                   Trixi.have_nonconservative_terms(equations_gpu),
                                   equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                               Trixi.have_nonconservative_terms(equations), equations,
                               solver.surface_integral, solver, cache)
    @test_approx (cache_gpu.elements.surface_flux_values,
                  cache.elements.surface_flux_values)

    TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu,
                                       equations_gpu, cache_gpu)
    Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
    @test_approx (cache_gpu.boundaries.u, cache.boundaries.u)

    TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                                  Trixi.have_nonconservative_terms(equations_gpu),
                                  equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                              solver.surface_integral, solver)
    @test_approx (cache_gpu.elements.surface_flux_values,
                  cache.elements.surface_flux_values)

    TrixiCUDA.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral,
                                 solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
    Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.cuda_sources!(du_gpu, u_gpu, t_gpu, source_terms_gpu,
                            equations_gpu, cache_gpu)
    Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
    @test_approx (du_gpu, du)
end

end # module
