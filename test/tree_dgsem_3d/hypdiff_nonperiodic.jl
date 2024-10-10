module TestHypdiffNonperiodic3D

include("../test_macros.jl")

@testset "Hypdiff Nonperiodic 3D" begin
    equations = HyperbolicDiffusionEquations3D()

    initial_condition = initial_condition_poisson_nonperiodic
    boundary_conditions = (x_neg = boundary_condition_poisson_nonperiodic,
                           x_pos = boundary_condition_poisson_nonperiodic,
                           y_neg = boundary_condition_periodic,
                           y_pos = boundary_condition_periodic,
                           z_neg = boundary_condition_periodic,
                           z_pos = boundary_condition_periodic)

    solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

    coordinates_min = (0.0, 0.0, 0.0)
    coordinates_max = (1.0, 1.0, 1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 2,
                    n_cells_max = 30_000,
                    periodicity = (false, true, true))

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        source_terms = source_terms_poisson_nonperiodic,
                                        boundary_conditions = boundary_conditions)
    semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver,
                                               source_terms = source_terms_poisson_nonperiodic,
                                               boundary_conditions = boundary_conditions)

    tspan = (0.0, 5.0)
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

    @testset "Components Initialization" begin
        @test_approx (u_gpu, u)
        # du is initlaizaed as undefined, cannot test now
    end

    @testset "Semidiscretization Process" begin
        @testset "Copy to GPU" begin
            du_gpu, u_gpu = TrixiCUDA.copy_to_gpu!(du, u)
            Trixi.reset_du!(du, solver, cache)
            @test_approx (du_gpu, du)
        end

        @testset "Volume Integral" begin
            TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                            Trixi.have_nonconservative_terms(equations_gpu),
                                            equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                            cache_gpu)
            Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                        equations, solver.volume_integral, solver, cache)
            @test_approx (du_gpu, du)
        end

        @testset "Prolong Interfaces" begin
            TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
            Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
            @test_approx (cache_gpu.interfaces.u, cache.interfaces.u)
        end

        @testset "Interface Flux" begin
            TrixiCUDA.cuda_interface_flux!(mesh_gpu,
                                           Trixi.have_nonconservative_terms(equations_gpu),
                                           equations_gpu, solver_gpu, cache_gpu)
            Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                       Trixi.have_nonconservative_terms(equations), equations,
                                       solver.surface_integral, solver, cache)
            @test_approx (cache_gpu.elements.surface_flux_values,
                          cache.elements.surface_flux_values)
        end

        @testset "Prolong Boundaries" begin
            TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu,
                                               equations_gpu, cache_gpu)
            Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
            @test_approx (cache_gpu.boundaries.u, cache.boundaries.u)
        end

        @testset "Boundary Flux" begin
            TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                                          Trixi.have_nonconservative_terms(equations_gpu),
                                          equations_gpu, solver_gpu, cache_gpu)
            Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                                      solver.surface_integral, solver)
            @test_approx (cache_gpu.elements.surface_flux_values,
                          cache.elements.surface_flux_values)
        end

        @testset "Prolong Mortars" begin
            TrixiCUDA.cuda_prolong2mortars!(u_gpu, mesh_gpu,
                                            TrixiCUDA.check_cache_mortars(cache_gpu),
                                            solver_gpu, cache_gpu)
            Trixi.prolong2mortars!(cache, u, mesh, equations,
                                   solver.mortar, solver.surface_integral, solver)
            @test_approx (cache_gpu.mortars.u_upper_left, cache.mortars.u_upper_left)
            @test_approx (cache_gpu.mortars.u_upper_right, cache.mortars.u_upper_right)
            @test_approx (cache_gpu.mortars.u_lower_left, cache.mortars.u_lower_left)
            @test_approx (cache_gpu.mortars.u_lower_right, cache.mortars.u_lower_right)
        end

        @testset "Mortar Flux" begin
            TrixiCUDA.cuda_mortar_flux!(mesh_gpu, TrixiCUDA.check_cache_mortars(cache_gpu),
                                        Trixi.have_nonconservative_terms(equations_gpu),
                                        equations_gpu, solver_gpu, cache_gpu)
            Trixi.calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                                    Trixi.have_nonconservative_terms(equations), equations,
                                    solver.mortar, solver.surface_integral, solver, cache)
            @test_approx (cache_gpu.elements.surface_flux_values,
                          cache.elements.surface_flux_values)
        end

        @testset "Surface Integral" begin
            TrixiCUDA.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
            Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral,
                                         solver, cache)
            @test_approx (du_gpu, du)
        end

        @testset "Apply Jacobian" begin
            TrixiCUDA.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
            Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
            @test_approx (du_gpu, du)
        end

        @testset "Apply Sources" begin
            TrixiCUDA.cuda_sources!(du_gpu, u_gpu, t_gpu, source_terms_gpu,
                                    equations_gpu, cache_gpu)
            Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
            @test_approx (du_gpu, du)
        end

        @testset "Copy to CPU" begin
            du_cpu, u_cpu = TrixiCUDA.copy_to_cpu!(du_gpu, u_gpu)
            @test_approx (du_cpu, du)
        end
    end
end

end # module
