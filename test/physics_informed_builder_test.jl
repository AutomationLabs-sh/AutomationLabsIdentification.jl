# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

module PhysicsInformedBuilderTests

using Test
using Dates
using Optim
using LineSearches
using ComputationalResources
using Flux

using AutomationLabsIdentification

import AutomationLabsIdentification: physics_informed_builder
import AutomationLabsIdentification: PHYSICS_INFORMED_LIST
import AutomationLabsIdentification: ALGORITHM_LIST

@testset "Physics informed builder" begin

    function QTP(du, u, p, t)
        #trainable parameters
        a1, a2, a3, a4 = p

        #constant parameters
        S = 0.06
        gamma_a = 0.3
        gamma_b = 0.4
        g = 9.81

        # a1 = 1.34e-4
        # a2 = 1.51e-4
        # a3 = 9.27e-5
        # a4 = 8.82e-5

        #states 
        x1 = u[1]
        x2 = u[2]
        x3 = u[3]
        x4 = u[4]
        qa = u[5]
        qb = u[6]

        du[1] =
            -a1 / S * sqrt(2 * g * x1) +
            a3 / S * sqrt(2 * g * x3) +
            gamma_a / (S * 3600) * qa
        du[2] =
            -a2 / S * sqrt(2 * g * x2) +
            a4 / S * sqrt(2 * g * x4) +
            gamma_b / (S * 3600) * qb
        du[3] = -a3 / S * sqrt(2 * g * x3) + (1 - gamma_b) / (S * 3600) * qb
        du[4] = -a4 / S * sqrt(2 * g * x4) + (1 - gamma_a) / (S * 3600) * qa
    end

    init_t_p = [1.1e-4, 1.2e-4, 9e-5, 9e-5]
    #init_t_p =#[1.34e-4, 1.51e-4, 9.27e-5, 8.82e-5]

    nbr_states = 4
    nbr_inputs = 2
    step_time = 5.0

    lower = [1e-5, 1e-5, 1e-5, 1e-5]
    upper = [10e-4, 10e-4, 10e-4, 10e-4]

    architecture = "physics_informed"
    solver = "lbfgs"

    sample_time = 5.0
    maximum_time = Dates.Minute(15)

    tuned_model_physics = physics_informed_builder(
        PHYSICS_INFORMED_LIST[Symbol(architecture)],
        QTP,
        ALGORITHM_LIST[Symbol(solver)],
        init_t_p,
        nbr_inputs,
        nbr_states,
        sample_time,
        maximum_time;
        lower_params = lower,
        upper_params = upper,
    ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_physics.model.optimiser) == LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}

    # Processors verification
    @test tuned_model_physics.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    # Parameters initialization
    @test tuned_model_physics.model.builder.t_p == init_t_p

    # nbr states and inputs 
    @test tuned_model_physics.model.builder.nbr_state == nbr_states
    @test tuned_model_physics.model.builder.nbr_input == nbr_inputs

    # Constraints
    @test tuned_model_physics.model.builder.constraints.lower == lower
    @test tuned_model_physics.model.builder.constraints.upper == upper

    # Sample time
    @test tuned_model_physics.model.builder.prob.tspan == (0.0, 5.0)

end

@testset "Physics informed and oracle builder" begin

    function QTP(du, u, p, t)
        #no trainable parameters
        #constant parameters
        S = 0.06
        gamma_a = 0.3
        gamma_b = 0.4
        g = 9.81

        a1 = 1.34e-4
        a2 = 1.51e-4
        a3 = 9.27e-5
        a4 = 8.82e-5

        #states 
        x1 = u[1]
        x2 = u[2]
        x3 = u[3]
        x4 = u[4]
        qa = u[5]
        qb = u[6]

        du[1] =
            -a1 / S * sqrt(2 * g * x1) +
            a3 / S * sqrt(2 * g * x3) +
            gamma_a / (S * 3600) * qa
        du[2] =
            -a2 / S * sqrt(2 * g * x2) +
            a4 / S * sqrt(2 * g * x4) +
            gamma_b / (S * 3600) * qb
        du[3] = -a3 / S * sqrt(2 * g * x3) + (1 - gamma_b) / (S * 3600) * qb
        du[4] = -a4 / S * sqrt(2 * g * x4) + (1 - gamma_a) / (S * 3600) * qa
    end

    init_t_p = [1.1e-4, 1.2e-4, 9e-5, 9e-5]
    #init_t_p =#[1.34e-4, 1.51e-4, 9.27e-5, 8.82e-5]
    nbr_states = 4
    nbr_inputs = 2
    sample_time = 5.0
    maximum_time = Dates.Minute(15)

    architecture = "physics_informed_oracle"
    solver = "lbfgs"

    tuned_model_physics_oracle = physics_informed_builder(
        PHYSICS_INFORMED_LIST[Symbol(architecture)],
        QTP,
        ALGORITHM_LIST[Symbol(solver)],
        init_t_p,
        nbr_inputs,
        nbr_states,
        sample_time,
        maximum_time
    ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_physics_oracle.model.optimiser) == LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}

    # Processors verification
    @test tuned_model_physics_oracle.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    # nbr states and inputs 
    @test tuned_model_physics_oracle.model.builder.physical.nbr_state == nbr_states
    @test tuned_model_physics_oracle.model.builder.physical.nbr_input == nbr_inputs
 
    # Sample time
    @test tuned_model_physics_oracle.model.builder.physical.prob.tspan == (0.0, 5.0)

    # Oracle Fnn
    @test tuned_model_physics_oracle.model.builder.oracle.neuron == 5
    @test tuned_model_physics_oracle.model.builder.oracle.layer == 2
    @test tuned_model_physics_oracle.model.builder.oracle.σ == Flux.relu

end

end