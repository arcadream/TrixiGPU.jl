module TestEulerBlobMortar2D

include("../test_macros.jl")

@testset "Euler Blob Mortar 2D" begin
    gamma = 5 / 3
    equations = CompressibleEulerEquations2D(gamma)

    function initial_condition_blob(x, t, equations::CompressibleEulerEquations2D)
        R = 1.0 # radius of the blob

        dens0 = 1.0
        Chi = 10.0 # density contrast

        tau_kh = 1.0
        tau_cr = tau_kh / 1.6 # crushing time

        velx0 = 2 * R * sqrt(Chi) / tau_cr
        vely0 = 0.0
        Ma0 = 2.7 # background flow Mach number Ma=v/c
        c = velx0 / Ma0 # sound speed

        p0 = c * c * dens0 / equations.gamma
        inicenter = SVector(-15, 0)
        x_rel = x - inicenter
        r = sqrt(x_rel[1]^2 + x_rel[2]^2)

        slope = 2
        dens = dens0 +
               (Chi - 1) * 0.5 * (1 + (tanh(slope * (r + R)) - (tanh(slope * (r - R)) + 1)))
        # velocity blob is zero
        velx = velx0 - velx0 * 0.5 * (1 + (tanh(slope * (r + R)) - (tanh(slope * (r - R)) + 1)))
        return prim2cons(SVector(dens, velx, vely0, p0), equations)
    end
    initial_condition = initial_condition_blob

    surface_flux = flux_lax_friedrichs
    volume_flux = flux_ranocha
    basis = LobattoLegendreBasis(3)

    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max = 0.05,
                                             alpha_min = 0.0001,
                                             alpha_smooth = true,
                                             variable = pressure)

    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg = volume_flux,
                                                     volume_flux_fv = surface_flux)

    solver = DGSEM(basis, surface_flux, volume_integral)

    coordinates_min = (-32.0, -32.0)
    coordinates_max = (32.0, 32.0)
    refinement_patches = ((type = "box", coordinates_min = (-40.0, -5.0),
                           coordinates_max = (40.0, 5.0)),
                          (type = "box", coordinates_min = (-40.0, -5.0),
                           coordinates_max = (40.0, 5.0)))
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 4,
                    refinement_patches = refinement_patches,
                    n_cells_max = 100_000)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
    semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver)

    tspan = (0.0, 16.0)
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
            Trixi.prolong2mortars!(cache, u, mesh, equations, solver.mortar,
                                   solver.surface_integral, solver)
            @test_approx (cache_gpu.mortars.u_upper, cache.mortars.u_upper)
            @test_approx (cache_gpu.mortars.u_lower, cache.mortars.u_lower)
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
