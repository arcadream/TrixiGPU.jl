module TestShallowWaterShock1D

using Trixi, TrixiCUDA
using Test

include("../test_macros.jl")

@testset "Shallow Water Shock 1D" begin
    equations = ShallowWaterEquations1D(gravity_constant = 9.812, H0 = 1.75)

    function initial_condition_stone_throw_discontinuous_bottom(x, t,
                                                                equations::ShallowWaterEquations1D)
        # Flat lake
        H = equations.H0

        # Discontinuous velocity
        v = 0.0
        if x[1] >= -0.75 && x[1] <= 0.0
            v = -1.0
        elseif x[1] >= 0.0 && x[1] <= 0.75
            v = 1.0
        end

        b = (1.5 / exp(0.5 * ((x[1] - 1.0)^2)) +
             0.75 / exp(0.5 * ((x[1] + 1.0)^2)))

        # Force a discontinuous bottom topography
        if x[1] >= -1.5 && x[1] <= 0.0
            b = 0.5
        end

        return prim2cons(SVector(H, v, b), equations)
    end

    initial_condition = initial_condition_stone_throw_discontinuous_bottom

    boundary_condition = boundary_condition_slip_wall

    volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
    surface_flux = (FluxHydrostaticReconstruction(flux_lax_friedrichs,
                                                  hydrostatic_reconstruction_audusse_etal),
                    flux_nonconservative_audusse_etal)
    basis = LobattoLegendreBasis(4)

    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max = 0.5,
                                             alpha_min = 0.001,
                                             alpha_smooth = true,
                                             variable = waterheight_pressure)
    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg = volume_flux,
                                                     volume_flux_fv = surface_flux)

    solver = DGSEM(basis, surface_flux, volume_integral)

    coordinates_min = -3.0
    coordinates_max = 3.0
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 3,
                    n_cells_max = 10_000,
                    periodicity = false)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions = boundary_condition)
    semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver,
                                               boundary_conditions = boundary_condition)

    tspan = (0.0, 3.0)
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
