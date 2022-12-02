# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
module NeulralNetworkBuilderTests

using Test
using Dates
using Flux
using Optim
using ComputationalResources
using LineSearches
using MLJMultivariateStatsInterface

using AutomationLabsIdentification

import AutomationLabsIdentification: _neural_network_builder
import AutomationLabsIdentification: ARCHITECTURE_LIST
import AutomationLabsIdentification: PROCESSOR_LIST
import AutomationLabsIdentification: ALGORITHM_LIST

@testset "Neural Network Builder linear modification algorithms" begin

    architecture = "linear"
    processor = "cpu_1"
    algorithm = "lls"
    maximum_time = Dates.Minute(15)

    tuned_model_linear_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    @test typeof(tuned_model_linear_0) == MLJMultivariateStatsInterface.MultitargetLinearRegressor
    
end

@testset "Neural Network Builder Fnn modification algorithms" begin

    architecture = "fnn"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_fnn_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_fnn_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_fnn_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_fnn_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_fnn_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_fnn_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_fnn_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    
    architecture = "fnn"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_fnn_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_fnn_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_fnn_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_fnn_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_fnn_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_fnn_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_fnn_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    architecture = "fnn"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_fnn_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_fnn_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_fnn_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_fnn_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_fnn_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_fnn_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_fnn_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_fnn_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_fnn_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_fnn_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_fnn_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_fnn_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_fnn_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_fnn_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_fnn_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_fnn_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_fnn_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_fnn_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_fnn_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_fnn_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_fnn_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_fnn_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_fnn_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_fnn_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_fnn_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_fnn_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_fnn_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_fnn_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_fnn_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_fnn_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_fnn_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_fnn_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_fnn_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_fnn_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_fnn_0.model.range[1].lower == 3
    @test tuned_model_fnn_0.model.range[1].upper == 10
    @test tuned_model_fnn_0.model.range[2].lower == 1
    @test tuned_model_fnn_0.model.range[2].upper == 6
    @test tuned_model_fnn_0.model.range[3].lower == 50
    @test tuned_model_fnn_0.model.range[3].upper == 500
    @test tuned_model_fnn_0.model.model.batch_size == 512
    @test tuned_model_fnn_0.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_1.model.range[1].lower == 3
    @test tuned_model_fnn_1.model.range[1].upper == 10
    @test tuned_model_fnn_1.model.range[2].lower == 1
    @test tuned_model_fnn_1.model.range[2].upper == 6
    @test tuned_model_fnn_1.model.range[3].lower == 50
    @test tuned_model_fnn_1.model.range[3].upper == 500
    @test tuned_model_fnn_1.model.model.batch_size == 512
    @test tuned_model_fnn_1.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_2.model.range[1].lower == 3
    @test tuned_model_fnn_2.model.range[1].upper == 10
    @test tuned_model_fnn_2.model.range[2].lower == 1
    @test tuned_model_fnn_2.model.range[2].upper == 6
    @test tuned_model_fnn_2.model.range[3].lower == 50
    @test tuned_model_fnn_2.model.range[3].upper == 500
    @test tuned_model_fnn_2.model.model.batch_size == 512
    @test tuned_model_fnn_2.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_3.model.range[1].lower == 3
    @test tuned_model_fnn_3.model.range[1].upper == 10
    @test tuned_model_fnn_3.model.range[2].lower == 1
    @test tuned_model_fnn_3.model.range[2].upper == 6
    @test tuned_model_fnn_3.model.range[3].lower == 50
    @test tuned_model_fnn_3.model.range[3].upper == 500
    @test tuned_model_fnn_3.model.model.batch_size == 512
    @test tuned_model_fnn_3.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_4.model.range[1].lower == 3
    @test tuned_model_fnn_4.model.range[1].upper == 10
    @test tuned_model_fnn_4.model.range[2].lower == 1
    @test tuned_model_fnn_4.model.range[2].upper == 6
    @test tuned_model_fnn_4.model.range[3].lower == 50
    @test tuned_model_fnn_4.model.range[3].upper == 500
    @test tuned_model_fnn_4.model.model.batch_size == 512
    @test tuned_model_fnn_4.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_5.model.range[1].lower == 3
    @test tuned_model_fnn_5.model.range[1].upper == 10
    @test tuned_model_fnn_5.model.range[2].lower == 1
    @test tuned_model_fnn_5.model.range[2].upper == 6
    @test tuned_model_fnn_5.model.range[3].lower == 50
    @test tuned_model_fnn_5.model.range[3].upper == 500
    @test tuned_model_fnn_5.model.model.batch_size == 512
    @test tuned_model_fnn_5.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_6.model.range[1].lower == 3
    @test tuned_model_fnn_6.model.range[1].upper == 10
    @test tuned_model_fnn_6.model.range[2].lower == 1
    @test tuned_model_fnn_6.model.range[2].upper == 6
    @test tuned_model_fnn_6.model.range[3].lower == 50
    @test tuned_model_fnn_6.model.range[3].upper == 500
    @test tuned_model_fnn_6.model.model.batch_size == 512
    @test tuned_model_fnn_6.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_7.model.range[1].lower == 3
    @test tuned_model_fnn_7.model.range[1].upper == 10
    @test tuned_model_fnn_7.model.range[2].lower == 1
    @test tuned_model_fnn_7.model.range[2].upper == 6
    @test tuned_model_fnn_7.model.range[3].lower == 50
    @test tuned_model_fnn_7.model.range[3].upper == 500
    @test tuned_model_fnn_7.model.model.batch_size == 512
    @test tuned_model_fnn_7.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_8.model.range[1].lower == 3
    @test tuned_model_fnn_8.model.range[1].upper == 10
    @test tuned_model_fnn_8.model.range[2].lower == 1
    @test tuned_model_fnn_8.model.range[2].upper == 6
    @test tuned_model_fnn_8.model.range[3].lower == 50
    @test tuned_model_fnn_8.model.range[3].upper == 500
    @test tuned_model_fnn_8.model.model.batch_size == 512
    @test tuned_model_fnn_8.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_9.model.range[1].lower == 3
    @test tuned_model_fnn_9.model.range[1].upper == 10
    @test tuned_model_fnn_9.model.range[2].lower == 1
    @test tuned_model_fnn_9.model.range[2].upper == 6
    @test tuned_model_fnn_9.model.range[3].lower == 50
    @test tuned_model_fnn_9.model.range[3].upper == 500
    @test tuned_model_fnn_9.model.model.batch_size == 512
    @test tuned_model_fnn_9.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_10.model.range[1].lower == 3
    @test tuned_model_fnn_10.model.range[1].upper == 10
    @test tuned_model_fnn_10.model.range[2].lower == 1
    @test tuned_model_fnn_10.model.range[2].upper == 6
    @test tuned_model_fnn_10.model.range[3].lower == 50
    @test tuned_model_fnn_10.model.range[3].upper == 500
    @test tuned_model_fnn_10.model.model.batch_size == 512
    @test tuned_model_fnn_10.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_11.model.range[1].lower == 3
    @test tuned_model_fnn_11.model.range[1].upper == 10
    @test tuned_model_fnn_11.model.range[2].lower == 1
    @test tuned_model_fnn_11.model.range[2].upper == 6
    @test tuned_model_fnn_11.model.range[3].lower == 50
    @test tuned_model_fnn_11.model.range[3].upper == 500
    @test tuned_model_fnn_11.model.model.batch_size == 512
    @test tuned_model_fnn_11.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_12.model.range[1].lower == 3
    @test tuned_model_fnn_12.model.range[1].upper == 10
    @test tuned_model_fnn_12.model.range[2].lower == 1
    @test tuned_model_fnn_12.model.range[2].upper == 6
    @test tuned_model_fnn_12.model.range[3].lower == 50
    @test tuned_model_fnn_12.model.range[3].upper == 500
    @test tuned_model_fnn_12.model.model.batch_size == 512
    @test tuned_model_fnn_12.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_13.model.range[1].lower == 3
    @test tuned_model_fnn_13.model.range[1].upper == 10
    @test tuned_model_fnn_13.model.range[2].lower == 1
    @test tuned_model_fnn_13.model.range[2].upper == 6
    @test tuned_model_fnn_13.model.range[3].lower == 50
    @test tuned_model_fnn_13.model.range[3].upper == 500
    @test tuned_model_fnn_13.model.model.batch_size == 512
    @test tuned_model_fnn_13.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_14.model.range[1].lower == 3
    @test tuned_model_fnn_14.model.range[1].upper == 10
    @test tuned_model_fnn_14.model.range[2].lower == 1
    @test tuned_model_fnn_14.model.range[2].upper == 6
    @test tuned_model_fnn_14.model.range[3].lower == 50
    @test tuned_model_fnn_14.model.range[3].upper == 500
    @test tuned_model_fnn_14.model.model.batch_size == 512
    @test tuned_model_fnn_14.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_15.model.range[1].lower == 3
    @test tuned_model_fnn_15.model.range[1].upper == 10
    @test tuned_model_fnn_15.model.range[2].lower == 1
    @test tuned_model_fnn_15.model.range[2].upper == 6
    @test tuned_model_fnn_15.model.range[3].lower == 50
    @test tuned_model_fnn_15.model.range[3].upper == 500
    @test tuned_model_fnn_15.model.model.batch_size == 512
    @test tuned_model_fnn_15.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_16.model.range[1].lower == 3
    @test tuned_model_fnn_16.model.range[1].upper == 10
    @test tuned_model_fnn_16.model.range[2].lower == 1
    @test tuned_model_fnn_16.model.range[2].upper == 6
    @test tuned_model_fnn_16.model.range[3].lower == 50
    @test tuned_model_fnn_16.model.range[3].upper == 500
    @test tuned_model_fnn_16.model.model.batch_size == 512
    @test tuned_model_fnn_16.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_17.model.range[1].lower == 3
    @test tuned_model_fnn_17.model.range[1].upper == 10
    @test tuned_model_fnn_17.model.range[2].lower == 1
    @test tuned_model_fnn_17.model.range[2].upper == 6
    @test tuned_model_fnn_17.model.range[3].lower == 50
    @test tuned_model_fnn_17.model.range[3].upper == 500
    @test tuned_model_fnn_17.model.model.batch_size == 512
    @test tuned_model_fnn_17.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_18.model.range[1].lower == 3
    @test tuned_model_fnn_18.model.range[1].upper == 10
    @test tuned_model_fnn_18.model.range[2].lower == 1
    @test tuned_model_fnn_18.model.range[2].upper == 6
    @test tuned_model_fnn_18.model.range[3].lower == 50
    @test tuned_model_fnn_18.model.range[3].upper == 500
    @test tuned_model_fnn_18.model.model.batch_size == 512
    @test tuned_model_fnn_18.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_19.model.range[1].lower == 3
    @test tuned_model_fnn_19.model.range[1].upper == 10
    @test tuned_model_fnn_19.model.range[2].lower == 1
    @test tuned_model_fnn_19.model.range[2].upper == 6
    @test tuned_model_fnn_19.model.range[3].lower == 50
    @test tuned_model_fnn_19.model.range[3].upper == 500
    @test tuned_model_fnn_19.model.model.batch_size == 512
    @test tuned_model_fnn_19.model.model.builder.σ == Flux.relu

    @test tuned_model_fnn_20.model.range[1].lower == 3
    @test tuned_model_fnn_20.model.range[1].upper == 10
    @test tuned_model_fnn_20.model.range[2].lower == 1
    @test tuned_model_fnn_20.model.range[2].upper == 6
    @test tuned_model_fnn_20.model.range[3].lower == 50
    @test tuned_model_fnn_20.model.range[3].upper == 500
    @test tuned_model_fnn_20.model.model.batch_size == 512
    @test tuned_model_fnn_20.model.model.builder.σ == Flux.relu

end


@testset "Neural Network Builder Icnn modification algorithms" begin

    architecture = "icnn"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_icnn_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_icnn_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_icnn_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_icnn_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_icnn_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_icnn_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_icnn_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    
    architecture = "icnn"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_icnn_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_icnn_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_icnn_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_icnn_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_icnn_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_icnn_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_icnn_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    architecture = "icnn"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_icnn_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_icnn_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_icnn_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_icnn_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_icnn_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_icnn_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_icnn_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_icnn_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_icnn_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_icnn_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_icnn_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_icnn_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_icnn_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_icnn_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_icnn_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_icnn_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_icnn_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_icnn_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_icnn_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_icnn_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_icnn_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_icnn_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_icnn_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_icnn_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_icnn_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_icnn_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_icnn_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_icnn_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_icnn_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_icnn_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_icnn_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_icnn_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_icnn_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_icnn_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_icnn_0.model.range[1].lower == 3
    @test tuned_model_icnn_0.model.range[1].upper == 10
    @test tuned_model_icnn_0.model.range[2].lower == 1
    @test tuned_model_icnn_0.model.range[2].upper == 6
    @test tuned_model_icnn_0.model.range[3].lower == 50
    @test tuned_model_icnn_0.model.range[3].upper == 500
    @test tuned_model_icnn_0.model.model.batch_size == 512
    @test tuned_model_icnn_0.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_1.model.range[1].lower == 3
    @test tuned_model_icnn_1.model.range[1].upper == 10
    @test tuned_model_icnn_1.model.range[2].lower == 1
    @test tuned_model_icnn_1.model.range[2].upper == 6
    @test tuned_model_icnn_1.model.range[3].lower == 50
    @test tuned_model_icnn_1.model.range[3].upper == 500
    @test tuned_model_icnn_1.model.model.batch_size == 512
    @test tuned_model_icnn_1.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_2.model.range[1].lower == 3
    @test tuned_model_icnn_2.model.range[1].upper == 10
    @test tuned_model_icnn_2.model.range[2].lower == 1
    @test tuned_model_icnn_2.model.range[2].upper == 6
    @test tuned_model_icnn_2.model.range[3].lower == 50
    @test tuned_model_icnn_2.model.range[3].upper == 500
    @test tuned_model_icnn_2.model.model.batch_size == 512
    @test tuned_model_icnn_2.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_3.model.range[1].lower == 3
    @test tuned_model_icnn_3.model.range[1].upper == 10
    @test tuned_model_icnn_3.model.range[2].lower == 1
    @test tuned_model_icnn_3.model.range[2].upper == 6
    @test tuned_model_icnn_3.model.range[3].lower == 50
    @test tuned_model_icnn_3.model.range[3].upper == 500
    @test tuned_model_icnn_3.model.model.batch_size == 512
    @test tuned_model_icnn_3.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_4.model.range[1].lower == 3
    @test tuned_model_icnn_4.model.range[1].upper == 10
    @test tuned_model_icnn_4.model.range[2].lower == 1
    @test tuned_model_icnn_4.model.range[2].upper == 6
    @test tuned_model_icnn_4.model.range[3].lower == 50
    @test tuned_model_icnn_4.model.range[3].upper == 500
    @test tuned_model_icnn_4.model.model.batch_size == 512
    @test tuned_model_icnn_4.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_5.model.range[1].lower == 3
    @test tuned_model_icnn_5.model.range[1].upper == 10
    @test tuned_model_icnn_5.model.range[2].lower == 1
    @test tuned_model_icnn_5.model.range[2].upper == 6
    @test tuned_model_icnn_5.model.range[3].lower == 50
    @test tuned_model_icnn_5.model.range[3].upper == 500
    @test tuned_model_icnn_5.model.model.batch_size == 512
    @test tuned_model_icnn_5.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_6.model.range[1].lower == 3
    @test tuned_model_icnn_6.model.range[1].upper == 10
    @test tuned_model_icnn_6.model.range[2].lower == 1
    @test tuned_model_icnn_6.model.range[2].upper == 6
    @test tuned_model_icnn_6.model.range[3].lower == 50
    @test tuned_model_icnn_6.model.range[3].upper == 500
    @test tuned_model_icnn_6.model.model.batch_size == 512
    @test tuned_model_icnn_6.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_7.model.range[1].lower == 3
    @test tuned_model_icnn_7.model.range[1].upper == 10
    @test tuned_model_icnn_7.model.range[2].lower == 1
    @test tuned_model_icnn_7.model.range[2].upper == 6
    @test tuned_model_icnn_7.model.range[3].lower == 50
    @test tuned_model_icnn_7.model.range[3].upper == 500
    @test tuned_model_icnn_7.model.model.batch_size == 512
    @test tuned_model_icnn_7.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_8.model.range[1].lower == 3
    @test tuned_model_icnn_8.model.range[1].upper == 10
    @test tuned_model_icnn_8.model.range[2].lower == 1
    @test tuned_model_icnn_8.model.range[2].upper == 6
    @test tuned_model_icnn_8.model.range[3].lower == 50
    @test tuned_model_icnn_8.model.range[3].upper == 500
    @test tuned_model_icnn_8.model.model.batch_size == 512
    @test tuned_model_icnn_8.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_9.model.range[1].lower == 3
    @test tuned_model_icnn_9.model.range[1].upper == 10
    @test tuned_model_icnn_9.model.range[2].lower == 1
    @test tuned_model_icnn_9.model.range[2].upper == 6
    @test tuned_model_icnn_9.model.range[3].lower == 50
    @test tuned_model_icnn_9.model.range[3].upper == 500
    @test tuned_model_icnn_9.model.model.batch_size == 512
    @test tuned_model_icnn_9.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_10.model.range[1].lower == 3
    @test tuned_model_icnn_10.model.range[1].upper == 10
    @test tuned_model_icnn_10.model.range[2].lower == 1
    @test tuned_model_icnn_10.model.range[2].upper == 6
    @test tuned_model_icnn_10.model.range[3].lower == 50
    @test tuned_model_icnn_10.model.range[3].upper == 500
    @test tuned_model_icnn_10.model.model.batch_size == 512
    @test tuned_model_icnn_10.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_11.model.range[1].lower == 3
    @test tuned_model_icnn_11.model.range[1].upper == 10
    @test tuned_model_icnn_11.model.range[2].lower == 1
    @test tuned_model_icnn_11.model.range[2].upper == 6
    @test tuned_model_icnn_11.model.range[3].lower == 50
    @test tuned_model_icnn_11.model.range[3].upper == 500
    @test tuned_model_icnn_11.model.model.batch_size == 512
    @test tuned_model_icnn_11.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_12.model.range[1].lower == 3
    @test tuned_model_icnn_12.model.range[1].upper == 10
    @test tuned_model_icnn_12.model.range[2].lower == 1
    @test tuned_model_icnn_12.model.range[2].upper == 6
    @test tuned_model_icnn_12.model.range[3].lower == 50
    @test tuned_model_icnn_12.model.range[3].upper == 500
    @test tuned_model_icnn_12.model.model.batch_size == 512
    @test tuned_model_icnn_12.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_13.model.range[1].lower == 3
    @test tuned_model_icnn_13.model.range[1].upper == 10
    @test tuned_model_icnn_13.model.range[2].lower == 1
    @test tuned_model_icnn_13.model.range[2].upper == 6
    @test tuned_model_icnn_13.model.range[3].lower == 50
    @test tuned_model_icnn_13.model.range[3].upper == 500
    @test tuned_model_icnn_13.model.model.batch_size == 512
    @test tuned_model_icnn_13.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_14.model.range[1].lower == 3
    @test tuned_model_icnn_14.model.range[1].upper == 10
    @test tuned_model_icnn_14.model.range[2].lower == 1
    @test tuned_model_icnn_14.model.range[2].upper == 6
    @test tuned_model_icnn_14.model.range[3].lower == 50
    @test tuned_model_icnn_14.model.range[3].upper == 500
    @test tuned_model_icnn_14.model.model.batch_size == 512
    @test tuned_model_icnn_14.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_15.model.range[1].lower == 3
    @test tuned_model_icnn_15.model.range[1].upper == 10
    @test tuned_model_icnn_15.model.range[2].lower == 1
    @test tuned_model_icnn_15.model.range[2].upper == 6
    @test tuned_model_icnn_15.model.range[3].lower == 50
    @test tuned_model_icnn_15.model.range[3].upper == 500
    @test tuned_model_icnn_15.model.model.batch_size == 512
    @test tuned_model_icnn_15.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_16.model.range[1].lower == 3
    @test tuned_model_icnn_16.model.range[1].upper == 10
    @test tuned_model_icnn_16.model.range[2].lower == 1
    @test tuned_model_icnn_16.model.range[2].upper == 6
    @test tuned_model_icnn_16.model.range[3].lower == 50
    @test tuned_model_icnn_16.model.range[3].upper == 500
    @test tuned_model_icnn_16.model.model.batch_size == 512
    @test tuned_model_icnn_16.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_17.model.range[1].lower == 3
    @test tuned_model_icnn_17.model.range[1].upper == 10
    @test tuned_model_icnn_17.model.range[2].lower == 1
    @test tuned_model_icnn_17.model.range[2].upper == 6
    @test tuned_model_icnn_17.model.range[3].lower == 50
    @test tuned_model_icnn_17.model.range[3].upper == 500
    @test tuned_model_icnn_17.model.model.batch_size == 512
    @test tuned_model_icnn_17.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_18.model.range[1].lower == 3
    @test tuned_model_icnn_18.model.range[1].upper == 10
    @test tuned_model_icnn_18.model.range[2].lower == 1
    @test tuned_model_icnn_18.model.range[2].upper == 6
    @test tuned_model_icnn_18.model.range[3].lower == 50
    @test tuned_model_icnn_18.model.range[3].upper == 500
    @test tuned_model_icnn_18.model.model.batch_size == 512
    @test tuned_model_icnn_18.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_19.model.range[1].lower == 3
    @test tuned_model_icnn_19.model.range[1].upper == 10
    @test tuned_model_icnn_19.model.range[2].lower == 1
    @test tuned_model_icnn_19.model.range[2].upper == 6
    @test tuned_model_icnn_19.model.range[3].lower == 50
    @test tuned_model_icnn_19.model.range[3].upper == 500
    @test tuned_model_icnn_19.model.model.batch_size == 512
    @test tuned_model_icnn_19.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_20.model.range[1].lower == 3
    @test tuned_model_icnn_20.model.range[1].upper == 10
    @test tuned_model_icnn_20.model.range[2].lower == 1
    @test tuned_model_icnn_20.model.range[2].upper == 6
    @test tuned_model_icnn_20.model.range[3].lower == 50
    @test tuned_model_icnn_20.model.range[3].upper == 500
    @test tuned_model_icnn_20.model.model.batch_size == 512
    @test tuned_model_icnn_20.model.model.builder.σ == Flux.relu

end


@testset "Neural Network Builder ResNet modification algorithms" begin

    architecture = "resnet"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_resnet_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_resnet_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_resnet_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_resnet_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_resnet_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_resnet_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_resnet_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    
    architecture = "resnet"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_resnet_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_resnet_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_resnet_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_resnet_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_resnet_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_resnet_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_resnet_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    architecture = "resnet"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_resnet_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_resnet_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_resnet_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_resnet_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_resnet_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_resnet_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_resnet_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_resnet_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_resnet_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_resnet_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_resnet_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_resnet_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_resnet_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_resnet_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_resnet_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_resnet_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_resnet_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_resnet_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_resnet_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_resnet_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_resnet_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_resnet_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_resnet_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_resnet_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_resnet_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_resnet_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_resnet_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_resnet_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_resnet_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_resnet_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_resnet_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_resnet_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_resnet_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_resnet_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_resnet_0.model.range[1].lower == 3
    @test tuned_model_resnet_0.model.range[1].upper == 10
    @test tuned_model_resnet_0.model.range[2].lower == 1
    @test tuned_model_resnet_0.model.range[2].upper == 6
    @test tuned_model_resnet_0.model.range[3].lower == 50
    @test tuned_model_resnet_0.model.range[3].upper == 500
    @test tuned_model_resnet_0.model.model.batch_size == 512
    @test tuned_model_resnet_0.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_1.model.range[1].lower == 3
    @test tuned_model_resnet_1.model.range[1].upper == 10
    @test tuned_model_resnet_1.model.range[2].lower == 1
    @test tuned_model_resnet_1.model.range[2].upper == 6
    @test tuned_model_resnet_1.model.range[3].lower == 50
    @test tuned_model_resnet_1.model.range[3].upper == 500
    @test tuned_model_resnet_1.model.model.batch_size == 512
    @test tuned_model_resnet_1.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_2.model.range[1].lower == 3
    @test tuned_model_resnet_2.model.range[1].upper == 10
    @test tuned_model_resnet_2.model.range[2].lower == 1
    @test tuned_model_resnet_2.model.range[2].upper == 6
    @test tuned_model_resnet_2.model.range[3].lower == 50
    @test tuned_model_resnet_2.model.range[3].upper == 500
    @test tuned_model_resnet_2.model.model.batch_size == 512
    @test tuned_model_resnet_2.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_3.model.range[1].lower == 3
    @test tuned_model_resnet_3.model.range[1].upper == 10
    @test tuned_model_resnet_3.model.range[2].lower == 1
    @test tuned_model_resnet_3.model.range[2].upper == 6
    @test tuned_model_resnet_3.model.range[3].lower == 50
    @test tuned_model_resnet_3.model.range[3].upper == 500
    @test tuned_model_resnet_3.model.model.batch_size == 512
    @test tuned_model_resnet_3.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_4.model.range[1].lower == 3
    @test tuned_model_resnet_4.model.range[1].upper == 10
    @test tuned_model_resnet_4.model.range[2].lower == 1
    @test tuned_model_resnet_4.model.range[2].upper == 6
    @test tuned_model_resnet_4.model.range[3].lower == 50
    @test tuned_model_resnet_4.model.range[3].upper == 500
    @test tuned_model_resnet_4.model.model.batch_size == 512
    @test tuned_model_resnet_4.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_5.model.range[1].lower == 3
    @test tuned_model_resnet_5.model.range[1].upper == 10
    @test tuned_model_resnet_5.model.range[2].lower == 1
    @test tuned_model_resnet_5.model.range[2].upper == 6
    @test tuned_model_resnet_5.model.range[3].lower == 50
    @test tuned_model_resnet_5.model.range[3].upper == 500
    @test tuned_model_resnet_5.model.model.batch_size == 512
    @test tuned_model_resnet_5.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_6.model.range[1].lower == 3
    @test tuned_model_resnet_6.model.range[1].upper == 10
    @test tuned_model_resnet_6.model.range[2].lower == 1
    @test tuned_model_resnet_6.model.range[2].upper == 6
    @test tuned_model_resnet_6.model.range[3].lower == 50
    @test tuned_model_resnet_6.model.range[3].upper == 500
    @test tuned_model_resnet_6.model.model.batch_size == 512
    @test tuned_model_resnet_6.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_7.model.range[1].lower == 3
    @test tuned_model_resnet_7.model.range[1].upper == 10
    @test tuned_model_resnet_7.model.range[2].lower == 1
    @test tuned_model_resnet_7.model.range[2].upper == 6
    @test tuned_model_resnet_7.model.range[3].lower == 50
    @test tuned_model_resnet_7.model.range[3].upper == 500
    @test tuned_model_resnet_7.model.model.batch_size == 512
    @test tuned_model_resnet_7.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_8.model.range[1].lower == 3
    @test tuned_model_resnet_8.model.range[1].upper == 10
    @test tuned_model_resnet_8.model.range[2].lower == 1
    @test tuned_model_resnet_8.model.range[2].upper == 6
    @test tuned_model_resnet_8.model.range[3].lower == 50
    @test tuned_model_resnet_8.model.range[3].upper == 500
    @test tuned_model_resnet_8.model.model.batch_size == 512
    @test tuned_model_resnet_8.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_9.model.range[1].lower == 3
    @test tuned_model_resnet_9.model.range[1].upper == 10
    @test tuned_model_resnet_9.model.range[2].lower == 1
    @test tuned_model_resnet_9.model.range[2].upper == 6
    @test tuned_model_resnet_9.model.range[3].lower == 50
    @test tuned_model_resnet_9.model.range[3].upper == 500
    @test tuned_model_resnet_9.model.model.batch_size == 512
    @test tuned_model_resnet_9.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_10.model.range[1].lower == 3
    @test tuned_model_resnet_10.model.range[1].upper == 10
    @test tuned_model_resnet_10.model.range[2].lower == 1
    @test tuned_model_resnet_10.model.range[2].upper == 6
    @test tuned_model_resnet_10.model.range[3].lower == 50
    @test tuned_model_resnet_10.model.range[3].upper == 500
    @test tuned_model_resnet_10.model.model.batch_size == 512
    @test tuned_model_resnet_10.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_11.model.range[1].lower == 3
    @test tuned_model_resnet_11.model.range[1].upper == 10
    @test tuned_model_resnet_11.model.range[2].lower == 1
    @test tuned_model_resnet_11.model.range[2].upper == 6
    @test tuned_model_resnet_11.model.range[3].lower == 50
    @test tuned_model_resnet_11.model.range[3].upper == 500
    @test tuned_model_resnet_11.model.model.batch_size == 512
    @test tuned_model_resnet_11.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_12.model.range[1].lower == 3
    @test tuned_model_resnet_12.model.range[1].upper == 10
    @test tuned_model_resnet_12.model.range[2].lower == 1
    @test tuned_model_resnet_12.model.range[2].upper == 6
    @test tuned_model_resnet_12.model.range[3].lower == 50
    @test tuned_model_resnet_12.model.range[3].upper == 500
    @test tuned_model_resnet_12.model.model.batch_size == 512
    @test tuned_model_resnet_12.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_13.model.range[1].lower == 3
    @test tuned_model_resnet_13.model.range[1].upper == 10
    @test tuned_model_resnet_13.model.range[2].lower == 1
    @test tuned_model_resnet_13.model.range[2].upper == 6
    @test tuned_model_resnet_13.model.range[3].lower == 50
    @test tuned_model_resnet_13.model.range[3].upper == 500
    @test tuned_model_resnet_13.model.model.batch_size == 512
    @test tuned_model_resnet_13.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_14.model.range[1].lower == 3
    @test tuned_model_resnet_14.model.range[1].upper == 10
    @test tuned_model_resnet_14.model.range[2].lower == 1
    @test tuned_model_resnet_14.model.range[2].upper == 6
    @test tuned_model_resnet_14.model.range[3].lower == 50
    @test tuned_model_resnet_14.model.range[3].upper == 500
    @test tuned_model_resnet_14.model.model.batch_size == 512
    @test tuned_model_resnet_14.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_15.model.range[1].lower == 3
    @test tuned_model_resnet_15.model.range[1].upper == 10
    @test tuned_model_resnet_15.model.range[2].lower == 1
    @test tuned_model_resnet_15.model.range[2].upper == 6
    @test tuned_model_resnet_15.model.range[3].lower == 50
    @test tuned_model_resnet_15.model.range[3].upper == 500
    @test tuned_model_resnet_15.model.model.batch_size == 512
    @test tuned_model_resnet_15.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_16.model.range[1].lower == 3
    @test tuned_model_resnet_16.model.range[1].upper == 10
    @test tuned_model_resnet_16.model.range[2].lower == 1
    @test tuned_model_resnet_16.model.range[2].upper == 6
    @test tuned_model_resnet_16.model.range[3].lower == 50
    @test tuned_model_resnet_16.model.range[3].upper == 500
    @test tuned_model_resnet_16.model.model.batch_size == 512
    @test tuned_model_resnet_16.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_17.model.range[1].lower == 3
    @test tuned_model_resnet_17.model.range[1].upper == 10
    @test tuned_model_resnet_17.model.range[2].lower == 1
    @test tuned_model_resnet_17.model.range[2].upper == 6
    @test tuned_model_resnet_17.model.range[3].lower == 50
    @test tuned_model_resnet_17.model.range[3].upper == 500
    @test tuned_model_resnet_17.model.model.batch_size == 512
    @test tuned_model_resnet_17.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_18.model.range[1].lower == 3
    @test tuned_model_resnet_18.model.range[1].upper == 10
    @test tuned_model_resnet_18.model.range[2].lower == 1
    @test tuned_model_resnet_18.model.range[2].upper == 6
    @test tuned_model_resnet_18.model.range[3].lower == 50
    @test tuned_model_resnet_18.model.range[3].upper == 500
    @test tuned_model_resnet_18.model.model.batch_size == 512
    @test tuned_model_resnet_18.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_19.model.range[1].lower == 3
    @test tuned_model_resnet_19.model.range[1].upper == 10
    @test tuned_model_resnet_19.model.range[2].lower == 1
    @test tuned_model_resnet_19.model.range[2].upper == 6
    @test tuned_model_resnet_19.model.range[3].lower == 50
    @test tuned_model_resnet_19.model.range[3].upper == 500
    @test tuned_model_resnet_19.model.model.batch_size == 512
    @test tuned_model_resnet_19.model.model.builder.σ == Flux.relu

    @test tuned_model_resnet_20.model.range[1].lower == 3
    @test tuned_model_resnet_20.model.range[1].upper == 10
    @test tuned_model_resnet_20.model.range[2].lower == 1
    @test tuned_model_resnet_20.model.range[2].upper == 6
    @test tuned_model_resnet_20.model.range[3].lower == 50
    @test tuned_model_resnet_20.model.range[3].upper == 500
    @test tuned_model_resnet_20.model.model.batch_size == 512
    @test tuned_model_resnet_20.model.model.builder.σ == Flux.relu

end


@testset "Neural Network Builder densenet modification algorithms" begin

    architecture = "densenet"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_densenet_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_densenet_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_densenet_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_densenet_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_densenet_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_densenet_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_densenet_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    
    architecture = "densenet"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_densenet_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_densenet_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_densenet_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_densenet_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_densenet_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_densenet_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_densenet_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    architecture = "densenet"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_densenet_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_densenet_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_densenet_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_densenet_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_densenet_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_densenet_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_densenet_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_densenet_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_densenet_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_densenet_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_densenet_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_densenet_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_densenet_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_densenet_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_densenet_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_densenet_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_densenet_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_densenet_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_densenet_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_densenet_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_densenet_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_densenet_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_densenet_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_densenet_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_densenet_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_densenet_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_densenet_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_densenet_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_densenet_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_densenet_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_densenet_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_densenet_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_densenet_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_densenet_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_densenet_0.model.range[1].lower == 3
    @test tuned_model_densenet_0.model.range[1].upper == 10
    @test tuned_model_densenet_0.model.range[2].lower == 1
    @test tuned_model_densenet_0.model.range[2].upper == 6
    @test tuned_model_densenet_0.model.range[3].lower == 50
    @test tuned_model_densenet_0.model.range[3].upper == 500
    @test tuned_model_densenet_0.model.model.batch_size == 512
    @test tuned_model_densenet_0.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_1.model.range[1].lower == 3
    @test tuned_model_densenet_1.model.range[1].upper == 10
    @test tuned_model_densenet_1.model.range[2].lower == 1
    @test tuned_model_densenet_1.model.range[2].upper == 6
    @test tuned_model_densenet_1.model.range[3].lower == 50
    @test tuned_model_densenet_1.model.range[3].upper == 500
    @test tuned_model_densenet_1.model.model.batch_size == 512
    @test tuned_model_densenet_1.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_2.model.range[1].lower == 3
    @test tuned_model_densenet_2.model.range[1].upper == 10
    @test tuned_model_densenet_2.model.range[2].lower == 1
    @test tuned_model_densenet_2.model.range[2].upper == 6
    @test tuned_model_densenet_2.model.range[3].lower == 50
    @test tuned_model_densenet_2.model.range[3].upper == 500
    @test tuned_model_densenet_2.model.model.batch_size == 512
    @test tuned_model_densenet_2.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_3.model.range[1].lower == 3
    @test tuned_model_densenet_3.model.range[1].upper == 10
    @test tuned_model_densenet_3.model.range[2].lower == 1
    @test tuned_model_densenet_3.model.range[2].upper == 6
    @test tuned_model_densenet_3.model.range[3].lower == 50
    @test tuned_model_densenet_3.model.range[3].upper == 500
    @test tuned_model_densenet_3.model.model.batch_size == 512
    @test tuned_model_densenet_3.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_4.model.range[1].lower == 3
    @test tuned_model_densenet_4.model.range[1].upper == 10
    @test tuned_model_densenet_4.model.range[2].lower == 1
    @test tuned_model_densenet_4.model.range[2].upper == 6
    @test tuned_model_densenet_4.model.range[3].lower == 50
    @test tuned_model_densenet_4.model.range[3].upper == 500
    @test tuned_model_densenet_4.model.model.batch_size == 512
    @test tuned_model_densenet_4.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_5.model.range[1].lower == 3
    @test tuned_model_densenet_5.model.range[1].upper == 10
    @test tuned_model_densenet_5.model.range[2].lower == 1
    @test tuned_model_densenet_5.model.range[2].upper == 6
    @test tuned_model_densenet_5.model.range[3].lower == 50
    @test tuned_model_densenet_5.model.range[3].upper == 500
    @test tuned_model_densenet_5.model.model.batch_size == 512
    @test tuned_model_densenet_5.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_6.model.range[1].lower == 3
    @test tuned_model_densenet_6.model.range[1].upper == 10
    @test tuned_model_densenet_6.model.range[2].lower == 1
    @test tuned_model_densenet_6.model.range[2].upper == 6
    @test tuned_model_densenet_6.model.range[3].lower == 50
    @test tuned_model_densenet_6.model.range[3].upper == 500
    @test tuned_model_densenet_6.model.model.batch_size == 512
    @test tuned_model_densenet_6.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_7.model.range[1].lower == 3
    @test tuned_model_densenet_7.model.range[1].upper == 10
    @test tuned_model_densenet_7.model.range[2].lower == 1
    @test tuned_model_densenet_7.model.range[2].upper == 6
    @test tuned_model_densenet_7.model.range[3].lower == 50
    @test tuned_model_densenet_7.model.range[3].upper == 500
    @test tuned_model_densenet_7.model.model.batch_size == 512
    @test tuned_model_densenet_7.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_8.model.range[1].lower == 3
    @test tuned_model_densenet_8.model.range[1].upper == 10
    @test tuned_model_densenet_8.model.range[2].lower == 1
    @test tuned_model_densenet_8.model.range[2].upper == 6
    @test tuned_model_densenet_8.model.range[3].lower == 50
    @test tuned_model_densenet_8.model.range[3].upper == 500
    @test tuned_model_densenet_8.model.model.batch_size == 512
    @test tuned_model_densenet_8.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_9.model.range[1].lower == 3
    @test tuned_model_densenet_9.model.range[1].upper == 10
    @test tuned_model_densenet_9.model.range[2].lower == 1
    @test tuned_model_densenet_9.model.range[2].upper == 6
    @test tuned_model_densenet_9.model.range[3].lower == 50
    @test tuned_model_densenet_9.model.range[3].upper == 500
    @test tuned_model_densenet_9.model.model.batch_size == 512
    @test tuned_model_densenet_9.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_10.model.range[1].lower == 3
    @test tuned_model_densenet_10.model.range[1].upper == 10
    @test tuned_model_densenet_10.model.range[2].lower == 1
    @test tuned_model_densenet_10.model.range[2].upper == 6
    @test tuned_model_densenet_10.model.range[3].lower == 50
    @test tuned_model_densenet_10.model.range[3].upper == 500
    @test tuned_model_densenet_10.model.model.batch_size == 512
    @test tuned_model_densenet_10.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_11.model.range[1].lower == 3
    @test tuned_model_densenet_11.model.range[1].upper == 10
    @test tuned_model_densenet_11.model.range[2].lower == 1
    @test tuned_model_densenet_11.model.range[2].upper == 6
    @test tuned_model_densenet_11.model.range[3].lower == 50
    @test tuned_model_densenet_11.model.range[3].upper == 500
    @test tuned_model_densenet_11.model.model.batch_size == 512
    @test tuned_model_densenet_11.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_12.model.range[1].lower == 3
    @test tuned_model_densenet_12.model.range[1].upper == 10
    @test tuned_model_densenet_12.model.range[2].lower == 1
    @test tuned_model_densenet_12.model.range[2].upper == 6
    @test tuned_model_densenet_12.model.range[3].lower == 50
    @test tuned_model_densenet_12.model.range[3].upper == 500
    @test tuned_model_densenet_12.model.model.batch_size == 512
    @test tuned_model_densenet_12.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_13.model.range[1].lower == 3
    @test tuned_model_densenet_13.model.range[1].upper == 10
    @test tuned_model_densenet_13.model.range[2].lower == 1
    @test tuned_model_densenet_13.model.range[2].upper == 6
    @test tuned_model_densenet_13.model.range[3].lower == 50
    @test tuned_model_densenet_13.model.range[3].upper == 500
    @test tuned_model_densenet_13.model.model.batch_size == 512
    @test tuned_model_densenet_13.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_14.model.range[1].lower == 3
    @test tuned_model_densenet_14.model.range[1].upper == 10
    @test tuned_model_densenet_14.model.range[2].lower == 1
    @test tuned_model_densenet_14.model.range[2].upper == 6
    @test tuned_model_densenet_14.model.range[3].lower == 50
    @test tuned_model_densenet_14.model.range[3].upper == 500
    @test tuned_model_densenet_14.model.model.batch_size == 512
    @test tuned_model_densenet_14.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_15.model.range[1].lower == 3
    @test tuned_model_densenet_15.model.range[1].upper == 10
    @test tuned_model_densenet_15.model.range[2].lower == 1
    @test tuned_model_densenet_15.model.range[2].upper == 6
    @test tuned_model_densenet_15.model.range[3].lower == 50
    @test tuned_model_densenet_15.model.range[3].upper == 500
    @test tuned_model_densenet_15.model.model.batch_size == 512
    @test tuned_model_densenet_15.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_16.model.range[1].lower == 3
    @test tuned_model_densenet_16.model.range[1].upper == 10
    @test tuned_model_densenet_16.model.range[2].lower == 1
    @test tuned_model_densenet_16.model.range[2].upper == 6
    @test tuned_model_densenet_16.model.range[3].lower == 50
    @test tuned_model_densenet_16.model.range[3].upper == 500
    @test tuned_model_densenet_16.model.model.batch_size == 512
    @test tuned_model_densenet_16.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_17.model.range[1].lower == 3
    @test tuned_model_densenet_17.model.range[1].upper == 10
    @test tuned_model_densenet_17.model.range[2].lower == 1
    @test tuned_model_densenet_17.model.range[2].upper == 6
    @test tuned_model_densenet_17.model.range[3].lower == 50
    @test tuned_model_densenet_17.model.range[3].upper == 500
    @test tuned_model_densenet_17.model.model.batch_size == 512
    @test tuned_model_densenet_17.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_18.model.range[1].lower == 3
    @test tuned_model_densenet_18.model.range[1].upper == 10
    @test tuned_model_densenet_18.model.range[2].lower == 1
    @test tuned_model_densenet_18.model.range[2].upper == 6
    @test tuned_model_densenet_18.model.range[3].lower == 50
    @test tuned_model_densenet_18.model.range[3].upper == 500
    @test tuned_model_densenet_18.model.model.batch_size == 512
    @test tuned_model_densenet_18.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_19.model.range[1].lower == 3
    @test tuned_model_densenet_19.model.range[1].upper == 10
    @test tuned_model_densenet_19.model.range[2].lower == 1
    @test tuned_model_densenet_19.model.range[2].upper == 6
    @test tuned_model_densenet_19.model.range[3].lower == 50
    @test tuned_model_densenet_19.model.range[3].upper == 500
    @test tuned_model_densenet_19.model.model.batch_size == 512
    @test tuned_model_densenet_19.model.model.builder.σ == Flux.relu

    @test tuned_model_densenet_20.model.range[1].lower == 3
    @test tuned_model_densenet_20.model.range[1].upper == 10
    @test tuned_model_densenet_20.model.range[2].lower == 1
    @test tuned_model_densenet_20.model.range[2].upper == 6
    @test tuned_model_densenet_20.model.range[3].lower == 50
    @test tuned_model_densenet_20.model.range[3].upper == 500
    @test tuned_model_densenet_20.model.model.batch_size == 512
    @test tuned_model_densenet_20.model.model.builder.σ == Flux.relu

end



@testset "Neural Network Builder polynet modification algorithms" begin

    architecture = "polynet"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_polynet_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_polynet_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_polynet_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_polynet_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_polynet_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_polynet_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_polynet_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    
    architecture = "polynet"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_polynet_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_polynet_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_polynet_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_polynet_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_polynet_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_polynet_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_polynet_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    architecture = "polynet"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_polynet_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_polynet_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_polynet_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_polynet_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_polynet_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_polynet_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_polynet_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_polynet_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_polynet_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_polynet_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_polynet_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_polynet_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_polynet_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_polynet_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_polynet_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_polynet_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_polynet_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_polynet_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_polynet_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_polynet_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_polynet_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_polynet_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_polynet_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_polynet_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_polynet_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_polynet_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_polynet_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_polynet_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_polynet_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_polynet_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_polynet_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_polynet_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_polynet_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_polynet_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_polynet_0.model.range[1].lower == 3
    @test tuned_model_polynet_0.model.range[1].upper == 10
    @test tuned_model_polynet_0.model.range[2].lower == 1
    @test tuned_model_polynet_0.model.range[2].upper == 6
    @test tuned_model_polynet_0.model.range[3].lower == 50
    @test tuned_model_polynet_0.model.range[3].upper == 500
    @test tuned_model_polynet_0.model.model.batch_size == 512
    @test tuned_model_polynet_0.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_1.model.range[1].lower == 3
    @test tuned_model_polynet_1.model.range[1].upper == 10
    @test tuned_model_polynet_1.model.range[2].lower == 1
    @test tuned_model_polynet_1.model.range[2].upper == 6
    @test tuned_model_polynet_1.model.range[3].lower == 50
    @test tuned_model_polynet_1.model.range[3].upper == 500
    @test tuned_model_polynet_1.model.model.batch_size == 512
    @test tuned_model_polynet_1.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_2.model.range[1].lower == 3
    @test tuned_model_polynet_2.model.range[1].upper == 10
    @test tuned_model_polynet_2.model.range[2].lower == 1
    @test tuned_model_polynet_2.model.range[2].upper == 6
    @test tuned_model_polynet_2.model.range[3].lower == 50
    @test tuned_model_polynet_2.model.range[3].upper == 500
    @test tuned_model_polynet_2.model.model.batch_size == 512
    @test tuned_model_polynet_2.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_3.model.range[1].lower == 3
    @test tuned_model_polynet_3.model.range[1].upper == 10
    @test tuned_model_polynet_3.model.range[2].lower == 1
    @test tuned_model_polynet_3.model.range[2].upper == 6
    @test tuned_model_polynet_3.model.range[3].lower == 50
    @test tuned_model_polynet_3.model.range[3].upper == 500
    @test tuned_model_polynet_3.model.model.batch_size == 512
    @test tuned_model_polynet_3.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_4.model.range[1].lower == 3
    @test tuned_model_polynet_4.model.range[1].upper == 10
    @test tuned_model_polynet_4.model.range[2].lower == 1
    @test tuned_model_polynet_4.model.range[2].upper == 6
    @test tuned_model_polynet_4.model.range[3].lower == 50
    @test tuned_model_polynet_4.model.range[3].upper == 500
    @test tuned_model_polynet_4.model.model.batch_size == 512
    @test tuned_model_polynet_4.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_5.model.range[1].lower == 3
    @test tuned_model_polynet_5.model.range[1].upper == 10
    @test tuned_model_polynet_5.model.range[2].lower == 1
    @test tuned_model_polynet_5.model.range[2].upper == 6
    @test tuned_model_polynet_5.model.range[3].lower == 50
    @test tuned_model_polynet_5.model.range[3].upper == 500
    @test tuned_model_polynet_5.model.model.batch_size == 512
    @test tuned_model_polynet_5.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_6.model.range[1].lower == 3
    @test tuned_model_polynet_6.model.range[1].upper == 10
    @test tuned_model_polynet_6.model.range[2].lower == 1
    @test tuned_model_polynet_6.model.range[2].upper == 6
    @test tuned_model_polynet_6.model.range[3].lower == 50
    @test tuned_model_polynet_6.model.range[3].upper == 500
    @test tuned_model_polynet_6.model.model.batch_size == 512
    @test tuned_model_polynet_6.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_7.model.range[1].lower == 3
    @test tuned_model_polynet_7.model.range[1].upper == 10
    @test tuned_model_polynet_7.model.range[2].lower == 1
    @test tuned_model_polynet_7.model.range[2].upper == 6
    @test tuned_model_polynet_7.model.range[3].lower == 50
    @test tuned_model_polynet_7.model.range[3].upper == 500
    @test tuned_model_polynet_7.model.model.batch_size == 512
    @test tuned_model_polynet_7.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_8.model.range[1].lower == 3
    @test tuned_model_polynet_8.model.range[1].upper == 10
    @test tuned_model_polynet_8.model.range[2].lower == 1
    @test tuned_model_polynet_8.model.range[2].upper == 6
    @test tuned_model_polynet_8.model.range[3].lower == 50
    @test tuned_model_polynet_8.model.range[3].upper == 500
    @test tuned_model_polynet_8.model.model.batch_size == 512
    @test tuned_model_polynet_8.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_9.model.range[1].lower == 3
    @test tuned_model_polynet_9.model.range[1].upper == 10
    @test tuned_model_polynet_9.model.range[2].lower == 1
    @test tuned_model_polynet_9.model.range[2].upper == 6
    @test tuned_model_polynet_9.model.range[3].lower == 50
    @test tuned_model_polynet_9.model.range[3].upper == 500
    @test tuned_model_polynet_9.model.model.batch_size == 512
    @test tuned_model_polynet_9.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_10.model.range[1].lower == 3
    @test tuned_model_polynet_10.model.range[1].upper == 10
    @test tuned_model_polynet_10.model.range[2].lower == 1
    @test tuned_model_polynet_10.model.range[2].upper == 6
    @test tuned_model_polynet_10.model.range[3].lower == 50
    @test tuned_model_polynet_10.model.range[3].upper == 500
    @test tuned_model_polynet_10.model.model.batch_size == 512
    @test tuned_model_polynet_10.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_11.model.range[1].lower == 3
    @test tuned_model_polynet_11.model.range[1].upper == 10
    @test tuned_model_polynet_11.model.range[2].lower == 1
    @test tuned_model_polynet_11.model.range[2].upper == 6
    @test tuned_model_polynet_11.model.range[3].lower == 50
    @test tuned_model_polynet_11.model.range[3].upper == 500
    @test tuned_model_polynet_11.model.model.batch_size == 512
    @test tuned_model_polynet_11.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_12.model.range[1].lower == 3
    @test tuned_model_polynet_12.model.range[1].upper == 10
    @test tuned_model_polynet_12.model.range[2].lower == 1
    @test tuned_model_polynet_12.model.range[2].upper == 6
    @test tuned_model_polynet_12.model.range[3].lower == 50
    @test tuned_model_polynet_12.model.range[3].upper == 500
    @test tuned_model_polynet_12.model.model.batch_size == 512
    @test tuned_model_polynet_12.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_13.model.range[1].lower == 3
    @test tuned_model_polynet_13.model.range[1].upper == 10
    @test tuned_model_polynet_13.model.range[2].lower == 1
    @test tuned_model_polynet_13.model.range[2].upper == 6
    @test tuned_model_polynet_13.model.range[3].lower == 50
    @test tuned_model_polynet_13.model.range[3].upper == 500
    @test tuned_model_polynet_13.model.model.batch_size == 512
    @test tuned_model_polynet_13.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_14.model.range[1].lower == 3
    @test tuned_model_polynet_14.model.range[1].upper == 10
    @test tuned_model_polynet_14.model.range[2].lower == 1
    @test tuned_model_polynet_14.model.range[2].upper == 6
    @test tuned_model_polynet_14.model.range[3].lower == 50
    @test tuned_model_polynet_14.model.range[3].upper == 500
    @test tuned_model_polynet_14.model.model.batch_size == 512
    @test tuned_model_polynet_14.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_15.model.range[1].lower == 3
    @test tuned_model_polynet_15.model.range[1].upper == 10
    @test tuned_model_polynet_15.model.range[2].lower == 1
    @test tuned_model_polynet_15.model.range[2].upper == 6
    @test tuned_model_polynet_15.model.range[3].lower == 50
    @test tuned_model_polynet_15.model.range[3].upper == 500
    @test tuned_model_polynet_15.model.model.batch_size == 512
    @test tuned_model_polynet_15.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_16.model.range[1].lower == 3
    @test tuned_model_polynet_16.model.range[1].upper == 10
    @test tuned_model_polynet_16.model.range[2].lower == 1
    @test tuned_model_polynet_16.model.range[2].upper == 6
    @test tuned_model_polynet_16.model.range[3].lower == 50
    @test tuned_model_polynet_16.model.range[3].upper == 500
    @test tuned_model_polynet_16.model.model.batch_size == 512
    @test tuned_model_polynet_16.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_17.model.range[1].lower == 3
    @test tuned_model_polynet_17.model.range[1].upper == 10
    @test tuned_model_polynet_17.model.range[2].lower == 1
    @test tuned_model_polynet_17.model.range[2].upper == 6
    @test tuned_model_polynet_17.model.range[3].lower == 50
    @test tuned_model_polynet_17.model.range[3].upper == 500
    @test tuned_model_polynet_17.model.model.batch_size == 512
    @test tuned_model_polynet_17.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_18.model.range[1].lower == 3
    @test tuned_model_polynet_18.model.range[1].upper == 10
    @test tuned_model_polynet_18.model.range[2].lower == 1
    @test tuned_model_polynet_18.model.range[2].upper == 6
    @test tuned_model_polynet_18.model.range[3].lower == 50
    @test tuned_model_polynet_18.model.range[3].upper == 500
    @test tuned_model_polynet_18.model.model.batch_size == 512
    @test tuned_model_polynet_18.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_19.model.range[1].lower == 3
    @test tuned_model_polynet_19.model.range[1].upper == 10
    @test tuned_model_polynet_19.model.range[2].lower == 1
    @test tuned_model_polynet_19.model.range[2].upper == 6
    @test tuned_model_polynet_19.model.range[3].lower == 50
    @test tuned_model_polynet_19.model.range[3].upper == 500
    @test tuned_model_polynet_19.model.model.batch_size == 512
    @test tuned_model_polynet_19.model.model.builder.σ == Flux.relu

    @test tuned_model_polynet_20.model.range[1].lower == 3
    @test tuned_model_polynet_20.model.range[1].upper == 10
    @test tuned_model_polynet_20.model.range[2].lower == 1
    @test tuned_model_polynet_20.model.range[2].upper == 6
    @test tuned_model_polynet_20.model.range[3].lower == 50
    @test tuned_model_polynet_20.model.range[3].upper == 500
    @test tuned_model_polynet_20.model.model.batch_size == 512
    @test tuned_model_polynet_20.model.model.builder.σ == Flux.relu

end


@testset "Neural Network Builder neuralnetODE type 1 modification algorithms" begin

    architecture = "neuralnet_ode_type1"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_neuralnetODE_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    
    architecture = "neuralnet_ode_type1"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_neuralnetODE_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    architecture = "neuralnet_ode_type1"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_neuralnetODE_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_neuralnetODE_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_neuralnetODE_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_neuralnetODE_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_neuralnetODE_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_neuralnetODE_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_neuralnetODE_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_neuralnetODE_0.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_0.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_0.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_0.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_0.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_0.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_0.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_0.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_1.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_1.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_1.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_1.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_1.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_1.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_1.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_1.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_2.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_2.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_2.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_2.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_2.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_2.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_2.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_2.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_3.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_3.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_3.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_3.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_3.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_3.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_3.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_3.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_4.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_4.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_4.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_4.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_4.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_4.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_4.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_4.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_5.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_5.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_5.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_5.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_5.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_5.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_5.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_5.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_6.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_6.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_6.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_6.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_6.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_6.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_6.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_6.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_7.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_7.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_7.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_7.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_7.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_7.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_7.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_7.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_8.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_8.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_8.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_8.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_8.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_8.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_8.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_8.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_9.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_9.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_9.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_9.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_9.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_9.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_9.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_9.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_10.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_10.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_10.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_10.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_10.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_10.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_10.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_10.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_11.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_11.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_11.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_11.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_11.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_11.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_11.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_11.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_12.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_12.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_12.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_12.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_12.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_12.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_12.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_12.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_13.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_13.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_13.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_13.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_13.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_13.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_13.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_13.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_14.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_14.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_14.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_14.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_14.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_14.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_14.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_14.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_15.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_15.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_15.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_15.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_15.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_15.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_15.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_15.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_16.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_16.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_16.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_16.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_16.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_16.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_16.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_16.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_17.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_17.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_17.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_17.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_17.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_17.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_17.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_17.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_18.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_18.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_18.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_18.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_18.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_18.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_18.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_18.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_19.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_19.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_19.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_19.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_19.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_19.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_19.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_19.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_20.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_20.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_20.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_20.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_20.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_20.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_20.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_20.model.model.builder.σ == Flux.relu

end



@testset "Neural Network Builder neuralnetODE type 2 modification algorithms" begin

    architecture = "neuralnet_ode_type2"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_neuralnetODE_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    
    architecture = "neuralnet_ode_type2"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_neuralnetODE_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    architecture = "neuralnet_ode_type2"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_neuralnetODE_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_neuralnetODE_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_neuralnetODE_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_neuralnetODE_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_neuralnetODE_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_neuralnetODE_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_neuralnetODE_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_neuralnetODE_0.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_0.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_0.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_0.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_0.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_0.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_0.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_0.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_1.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_1.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_1.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_1.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_1.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_1.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_1.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_1.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_2.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_2.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_2.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_2.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_2.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_2.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_2.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_2.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_3.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_3.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_3.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_3.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_3.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_3.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_3.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_3.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_4.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_4.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_4.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_4.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_4.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_4.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_4.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_4.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_5.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_5.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_5.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_5.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_5.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_5.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_5.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_5.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_6.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_6.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_6.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_6.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_6.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_6.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_6.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_6.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_7.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_7.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_7.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_7.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_7.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_7.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_7.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_7.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_8.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_8.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_8.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_8.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_8.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_8.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_8.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_8.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_9.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_9.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_9.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_9.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_9.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_9.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_9.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_9.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_10.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_10.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_10.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_10.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_10.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_10.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_10.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_10.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_11.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_11.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_11.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_11.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_11.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_11.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_11.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_11.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_12.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_12.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_12.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_12.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_12.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_12.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_12.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_12.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_13.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_13.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_13.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_13.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_13.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_13.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_13.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_13.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_14.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_14.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_14.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_14.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_14.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_14.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_14.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_14.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_15.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_15.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_15.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_15.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_15.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_15.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_15.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_15.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_16.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_16.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_16.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_16.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_16.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_16.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_16.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_16.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_17.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_17.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_17.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_17.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_17.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_17.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_17.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_17.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_18.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_18.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_18.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_18.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_18.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_18.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_18.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_18.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_19.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_19.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_19.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_19.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_19.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_19.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_19.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_19.model.model.builder.σ == Flux.relu

    @test tuned_model_neuralnetODE_20.model.range[1].lower == 3
    @test tuned_model_neuralnetODE_20.model.range[1].upper == 10
    @test tuned_model_neuralnetODE_20.model.range[2].lower == 1
    @test tuned_model_neuralnetODE_20.model.range[2].upper == 6
    @test tuned_model_neuralnetODE_20.model.range[3].lower == 50
    @test tuned_model_neuralnetODE_20.model.range[3].upper == 500
    @test tuned_model_neuralnetODE_20.model.model.batch_size == 512
    @test tuned_model_neuralnetODE_20.model.model.builder.σ == Flux.relu

end

@testset "Neural Network Builder rbf modification algorithms" begin

    architecture = "rbf"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_rbf_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_rbf_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_rbf_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_rbf_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rbf_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rbf_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_rbf_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    
    architecture = "rbf"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_rbf_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_rbf_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_rbf_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_rbf_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rbf_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rbf_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_rbf_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    architecture = "rbf"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_rbf_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_rbf_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_rbf_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_rbf_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rbf_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rbf_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_rbf_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_rbf_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rbf_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rbf_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rbf_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rbf_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rbf_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rbf_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_rbf_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rbf_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rbf_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rbf_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rbf_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rbf_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rbf_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_rbf_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rbf_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rbf_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rbf_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rbf_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rbf_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rbf_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_rbf_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_rbf_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_rbf_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_rbf_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_rbf_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_rbf_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_rbf_0.model.range[1].lower == 3
    @test tuned_model_rbf_0.model.range[1].upper == 10
    @test tuned_model_rbf_0.model.range[2].lower == 50
    @test tuned_model_rbf_0.model.range[2].upper == 500
    @test tuned_model_rbf_0.model.model.batch_size == 512

    @test tuned_model_rbf_1.model.range[1].lower == 3
    @test tuned_model_rbf_1.model.range[1].upper == 10
    @test tuned_model_rbf_1.model.range[2].lower == 50
    @test tuned_model_rbf_1.model.range[2].upper == 500
    @test tuned_model_rbf_1.model.model.batch_size == 512

    @test tuned_model_rbf_2.model.range[1].lower == 3
    @test tuned_model_rbf_2.model.range[1].upper == 10
    @test tuned_model_rbf_2.model.range[2].lower == 50
    @test tuned_model_rbf_2.model.range[2].upper == 500
    @test tuned_model_rbf_2.model.model.batch_size == 512

    @test tuned_model_rbf_3.model.range[1].lower == 3
    @test tuned_model_rbf_3.model.range[1].upper == 10
    @test tuned_model_rbf_3.model.range[2].lower == 50
    @test tuned_model_rbf_3.model.range[2].upper == 500
    @test tuned_model_rbf_3.model.model.batch_size == 512

    @test tuned_model_rbf_4.model.range[1].lower == 3
    @test tuned_model_rbf_4.model.range[1].upper == 10
    @test tuned_model_rbf_4.model.range[2].lower == 50
    @test tuned_model_rbf_4.model.range[2].upper == 500
    @test tuned_model_rbf_4.model.model.batch_size == 512

    @test tuned_model_rbf_5.model.range[1].lower == 3
    @test tuned_model_rbf_5.model.range[1].upper == 10
    @test tuned_model_rbf_5.model.range[2].lower == 50
    @test tuned_model_rbf_5.model.range[2].upper == 500
    @test tuned_model_rbf_5.model.model.batch_size == 512

    @test tuned_model_rbf_6.model.range[1].lower == 3
    @test tuned_model_rbf_6.model.range[1].upper == 10
    @test tuned_model_rbf_6.model.range[2].lower == 50
    @test tuned_model_rbf_6.model.range[2].upper == 500
    @test tuned_model_rbf_6.model.model.batch_size == 512

    @test tuned_model_rbf_7.model.range[1].lower == 3
    @test tuned_model_rbf_7.model.range[1].upper == 10
    @test tuned_model_rbf_7.model.range[2].lower == 50
    @test tuned_model_rbf_7.model.range[2].upper == 500
    @test tuned_model_rbf_7.model.model.batch_size == 512

    @test tuned_model_rbf_8.model.range[1].lower == 3
    @test tuned_model_rbf_8.model.range[1].upper == 10
    @test tuned_model_rbf_8.model.range[2].lower == 50
    @test tuned_model_rbf_8.model.range[2].upper == 500
    @test tuned_model_rbf_8.model.model.batch_size == 512

    @test tuned_model_rbf_9.model.range[1].lower == 3
    @test tuned_model_rbf_9.model.range[1].upper == 10
    @test tuned_model_rbf_9.model.range[2].lower == 50
    @test tuned_model_rbf_9.model.range[2].upper == 500
    @test tuned_model_rbf_9.model.model.batch_size == 512

    @test tuned_model_rbf_10.model.range[1].lower == 3
    @test tuned_model_rbf_10.model.range[1].upper == 10
    @test tuned_model_rbf_10.model.range[2].lower == 50
    @test tuned_model_rbf_10.model.range[2].upper == 500
    @test tuned_model_rbf_10.model.model.batch_size == 512

    @test tuned_model_rbf_11.model.range[1].lower == 3
    @test tuned_model_rbf_11.model.range[1].upper == 10
    @test tuned_model_rbf_11.model.range[2].lower == 50
    @test tuned_model_rbf_11.model.range[2].upper == 500
    @test tuned_model_rbf_11.model.model.batch_size == 512

    @test tuned_model_rbf_12.model.range[1].lower == 3
    @test tuned_model_rbf_12.model.range[1].upper == 10
    @test tuned_model_rbf_12.model.range[2].lower == 50
    @test tuned_model_rbf_12.model.range[2].upper == 500
    @test tuned_model_rbf_12.model.model.batch_size == 512

    @test tuned_model_rbf_13.model.range[1].lower == 3
    @test tuned_model_rbf_13.model.range[1].upper == 10
    @test tuned_model_rbf_13.model.range[2].lower == 50
    @test tuned_model_rbf_13.model.range[2].upper == 500
    @test tuned_model_rbf_13.model.model.batch_size == 512

    @test tuned_model_rbf_14.model.range[1].lower == 3
    @test tuned_model_rbf_14.model.range[1].upper == 10
    @test tuned_model_rbf_14.model.range[2].lower == 50
    @test tuned_model_rbf_14.model.range[2].upper == 500
    @test tuned_model_rbf_14.model.model.batch_size == 512

    @test tuned_model_rbf_15.model.range[1].lower == 3
    @test tuned_model_rbf_15.model.range[1].upper == 10
    @test tuned_model_rbf_15.model.range[2].lower == 50
    @test tuned_model_rbf_15.model.range[2].upper == 500
    @test tuned_model_rbf_15.model.model.batch_size == 512

    @test tuned_model_rbf_16.model.range[1].lower == 3
    @test tuned_model_rbf_16.model.range[1].upper == 10
    @test tuned_model_rbf_16.model.range[2].lower == 50
    @test tuned_model_rbf_16.model.range[2].upper == 500
    @test tuned_model_rbf_16.model.model.batch_size == 512

    @test tuned_model_rbf_17.model.range[1].lower == 3
    @test tuned_model_rbf_17.model.range[1].upper == 10
    @test tuned_model_rbf_17.model.range[2].lower == 50
    @test tuned_model_rbf_17.model.range[2].upper == 500
    @test tuned_model_rbf_17.model.model.batch_size == 512

    @test tuned_model_rbf_18.model.range[1].lower == 3
    @test tuned_model_rbf_18.model.range[1].upper == 10
    @test tuned_model_rbf_18.model.range[2].lower == 50
    @test tuned_model_rbf_18.model.range[2].upper == 500
    @test tuned_model_rbf_18.model.model.batch_size == 512

    @test tuned_model_rbf_19.model.range[1].lower == 3
    @test tuned_model_rbf_19.model.range[1].upper == 10
    @test tuned_model_rbf_19.model.range[2].lower == 50
    @test tuned_model_rbf_19.model.range[2].upper == 500
    @test tuned_model_rbf_19.model.model.batch_size == 512

    @test tuned_model_rbf_20.model.range[1].lower == 3
    @test tuned_model_rbf_20.model.range[1].upper == 10
    @test tuned_model_rbf_20.model.range[2].lower == 50
    @test tuned_model_rbf_20.model.range[2].upper == 500
    @test tuned_model_rbf_20.model.model.batch_size == 512

end


@testset "Neural Network Builder rnn modification algorithms" begin

    architecture = "rnn"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_rnn_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_rnn_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_rnn_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_rnn_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rnn_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rnn_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_rnn_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    
    architecture = "rnn"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_rnn_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_rnn_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_rnn_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_rnn_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rnn_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rnn_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_rnn_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    architecture = "rnn"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_rnn_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_rnn_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_rnn_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_rnn_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rnn_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rnn_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_rnn_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_rnn_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rnn_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rnn_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rnn_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rnn_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rnn_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rnn_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_rnn_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rnn_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rnn_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rnn_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rnn_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rnn_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rnn_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_rnn_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rnn_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rnn_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rnn_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rnn_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rnn_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rnn_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_rnn_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_rnn_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_rnn_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_rnn_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_rnn_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_rnn_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_rnn_0.model.range[1].lower == 3
    @test tuned_model_rnn_0.model.range[1].upper == 10
    @test tuned_model_rnn_0.model.range[2].lower == 1
    @test tuned_model_rnn_0.model.range[2].upper == 6
    @test tuned_model_rnn_0.model.range[3].lower == 50
    @test tuned_model_rnn_0.model.range[3].upper == 500
    @test tuned_model_rnn_0.model.model.batch_size == 512
    @test tuned_model_rnn_0.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_1.model.range[1].lower == 3
    @test tuned_model_rnn_1.model.range[1].upper == 10
    @test tuned_model_rnn_1.model.range[2].lower == 1
    @test tuned_model_rnn_1.model.range[2].upper == 6
    @test tuned_model_rnn_1.model.range[3].lower == 50
    @test tuned_model_rnn_1.model.range[3].upper == 500
    @test tuned_model_rnn_1.model.model.batch_size == 512
    @test tuned_model_rnn_1.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_2.model.range[1].lower == 3
    @test tuned_model_rnn_2.model.range[1].upper == 10
    @test tuned_model_rnn_2.model.range[2].lower == 1
    @test tuned_model_rnn_2.model.range[2].upper == 6
    @test tuned_model_rnn_2.model.range[3].lower == 50
    @test tuned_model_rnn_2.model.range[3].upper == 500
    @test tuned_model_rnn_2.model.model.batch_size == 512
    @test tuned_model_rnn_2.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_3.model.range[1].lower == 3
    @test tuned_model_rnn_3.model.range[1].upper == 10
    @test tuned_model_rnn_3.model.range[2].lower == 1
    @test tuned_model_rnn_3.model.range[2].upper == 6
    @test tuned_model_rnn_3.model.range[3].lower == 50
    @test tuned_model_rnn_3.model.range[3].upper == 500
    @test tuned_model_rnn_3.model.model.batch_size == 512
    @test tuned_model_rnn_3.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_4.model.range[1].lower == 3
    @test tuned_model_rnn_4.model.range[1].upper == 10
    @test tuned_model_rnn_4.model.range[2].lower == 1
    @test tuned_model_rnn_4.model.range[2].upper == 6
    @test tuned_model_rnn_4.model.range[3].lower == 50
    @test tuned_model_rnn_4.model.range[3].upper == 500
    @test tuned_model_rnn_4.model.model.batch_size == 512
    @test tuned_model_rnn_4.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_5.model.range[1].lower == 3
    @test tuned_model_rnn_5.model.range[1].upper == 10
    @test tuned_model_rnn_5.model.range[2].lower == 1
    @test tuned_model_rnn_5.model.range[2].upper == 6
    @test tuned_model_rnn_5.model.range[3].lower == 50
    @test tuned_model_rnn_5.model.range[3].upper == 500
    @test tuned_model_rnn_5.model.model.batch_size == 512
    @test tuned_model_rnn_5.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_6.model.range[1].lower == 3
    @test tuned_model_rnn_6.model.range[1].upper == 10
    @test tuned_model_rnn_6.model.range[2].lower == 1
    @test tuned_model_rnn_6.model.range[2].upper == 6
    @test tuned_model_rnn_6.model.range[3].lower == 50
    @test tuned_model_rnn_6.model.range[3].upper == 500
    @test tuned_model_rnn_6.model.model.batch_size == 512
    @test tuned_model_rnn_6.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_7.model.range[1].lower == 3
    @test tuned_model_rnn_7.model.range[1].upper == 10
    @test tuned_model_rnn_7.model.range[2].lower == 1
    @test tuned_model_rnn_7.model.range[2].upper == 6
    @test tuned_model_rnn_7.model.range[3].lower == 50
    @test tuned_model_rnn_7.model.range[3].upper == 500
    @test tuned_model_rnn_7.model.model.batch_size == 512
    @test tuned_model_rnn_7.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_8.model.range[1].lower == 3
    @test tuned_model_rnn_8.model.range[1].upper == 10
    @test tuned_model_rnn_8.model.range[2].lower == 1
    @test tuned_model_rnn_8.model.range[2].upper == 6
    @test tuned_model_rnn_8.model.range[3].lower == 50
    @test tuned_model_rnn_8.model.range[3].upper == 500
    @test tuned_model_rnn_8.model.model.batch_size == 512
    @test tuned_model_rnn_8.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_9.model.range[1].lower == 3
    @test tuned_model_rnn_9.model.range[1].upper == 10
    @test tuned_model_rnn_9.model.range[2].lower == 1
    @test tuned_model_rnn_9.model.range[2].upper == 6
    @test tuned_model_rnn_9.model.range[3].lower == 50
    @test tuned_model_rnn_9.model.range[3].upper == 500
    @test tuned_model_rnn_9.model.model.batch_size == 512
    @test tuned_model_rnn_9.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_10.model.range[1].lower == 3
    @test tuned_model_rnn_10.model.range[1].upper == 10
    @test tuned_model_rnn_10.model.range[2].lower == 1
    @test tuned_model_rnn_10.model.range[2].upper == 6
    @test tuned_model_rnn_10.model.range[3].lower == 50
    @test tuned_model_rnn_10.model.range[3].upper == 500
    @test tuned_model_rnn_10.model.model.batch_size == 512
    @test tuned_model_rnn_10.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_11.model.range[1].lower == 3
    @test tuned_model_rnn_11.model.range[1].upper == 10
    @test tuned_model_rnn_11.model.range[2].lower == 1
    @test tuned_model_rnn_11.model.range[2].upper == 6
    @test tuned_model_rnn_11.model.range[3].lower == 50
    @test tuned_model_rnn_11.model.range[3].upper == 500
    @test tuned_model_rnn_11.model.model.batch_size == 512
    @test tuned_model_rnn_11.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_12.model.range[1].lower == 3
    @test tuned_model_rnn_12.model.range[1].upper == 10
    @test tuned_model_rnn_12.model.range[2].lower == 1
    @test tuned_model_rnn_12.model.range[2].upper == 6
    @test tuned_model_rnn_12.model.range[3].lower == 50
    @test tuned_model_rnn_12.model.range[3].upper == 500
    @test tuned_model_rnn_12.model.model.batch_size == 512
    @test tuned_model_rnn_12.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_13.model.range[1].lower == 3
    @test tuned_model_rnn_13.model.range[1].upper == 10
    @test tuned_model_rnn_13.model.range[2].lower == 1
    @test tuned_model_rnn_13.model.range[2].upper == 6
    @test tuned_model_rnn_13.model.range[3].lower == 50
    @test tuned_model_rnn_13.model.range[3].upper == 500
    @test tuned_model_rnn_13.model.model.batch_size == 512
    @test tuned_model_rnn_13.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_14.model.range[1].lower == 3
    @test tuned_model_rnn_14.model.range[1].upper == 10
    @test tuned_model_rnn_14.model.range[2].lower == 1
    @test tuned_model_rnn_14.model.range[2].upper == 6
    @test tuned_model_rnn_14.model.range[3].lower == 50
    @test tuned_model_rnn_14.model.range[3].upper == 500
    @test tuned_model_rnn_14.model.model.batch_size == 512
    @test tuned_model_rnn_14.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_15.model.range[1].lower == 3
    @test tuned_model_rnn_15.model.range[1].upper == 10
    @test tuned_model_rnn_15.model.range[2].lower == 1
    @test tuned_model_rnn_15.model.range[2].upper == 6
    @test tuned_model_rnn_15.model.range[3].lower == 50
    @test tuned_model_rnn_15.model.range[3].upper == 500
    @test tuned_model_rnn_15.model.model.batch_size == 512
    @test tuned_model_rnn_15.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_16.model.range[1].lower == 3
    @test tuned_model_rnn_16.model.range[1].upper == 10
    @test tuned_model_rnn_16.model.range[2].lower == 1
    @test tuned_model_rnn_16.model.range[2].upper == 6
    @test tuned_model_rnn_16.model.range[3].lower == 50
    @test tuned_model_rnn_16.model.range[3].upper == 500
    @test tuned_model_rnn_16.model.model.batch_size == 512
    @test tuned_model_rnn_16.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_17.model.range[1].lower == 3
    @test tuned_model_rnn_17.model.range[1].upper == 10
    @test tuned_model_rnn_17.model.range[2].lower == 1
    @test tuned_model_rnn_17.model.range[2].upper == 6
    @test tuned_model_rnn_17.model.range[3].lower == 50
    @test tuned_model_rnn_17.model.range[3].upper == 500
    @test tuned_model_rnn_17.model.model.batch_size == 512
    @test tuned_model_rnn_17.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_18.model.range[1].lower == 3
    @test tuned_model_rnn_18.model.range[1].upper == 10
    @test tuned_model_rnn_18.model.range[2].lower == 1
    @test tuned_model_rnn_18.model.range[2].upper == 6
    @test tuned_model_rnn_18.model.range[3].lower == 50
    @test tuned_model_rnn_18.model.range[3].upper == 500
    @test tuned_model_rnn_18.model.model.batch_size == 512
    @test tuned_model_rnn_18.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_19.model.range[1].lower == 3
    @test tuned_model_rnn_19.model.range[1].upper == 10
    @test tuned_model_rnn_19.model.range[2].lower == 1
    @test tuned_model_rnn_19.model.range[2].upper == 6
    @test tuned_model_rnn_19.model.range[3].lower == 50
    @test tuned_model_rnn_19.model.range[3].upper == 500
    @test tuned_model_rnn_19.model.model.batch_size == 512
    @test tuned_model_rnn_19.model.model.builder.σ == Flux.relu

    @test tuned_model_rnn_20.model.range[1].lower == 3
    @test tuned_model_rnn_20.model.range[1].upper == 10
    @test tuned_model_rnn_20.model.range[2].lower == 1
    @test tuned_model_rnn_20.model.range[2].upper == 6
    @test tuned_model_rnn_20.model.range[3].lower == 50
    @test tuned_model_rnn_20.model.range[3].upper == 500
    @test tuned_model_rnn_20.model.model.batch_size == 512
    @test tuned_model_rnn_20.model.model.builder.σ == Flux.relu

end


@testset "Neural Network Builder lstm modification algorithms" begin

    architecture = "lstm"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_lstm_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_lstm_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_lstm_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_lstm_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_lstm_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_lstm_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_lstm_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    
    architecture = "lstm"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_lstm_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_lstm_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_lstm_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_lstm_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_lstm_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_lstm_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_lstm_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    architecture = "lstm"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_lstm_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_lstm_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_lstm_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_lstm_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_lstm_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_lstm_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_lstm_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_lstm_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_lstm_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_lstm_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_lstm_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_lstm_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_lstm_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_lstm_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_lstm_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_lstm_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_lstm_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_lstm_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_lstm_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_lstm_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_lstm_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_lstm_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_lstm_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_lstm_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_lstm_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_lstm_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_lstm_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_lstm_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_lstm_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_lstm_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_lstm_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_lstm_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_lstm_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_lstm_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_lstm_0.model.range[1].lower == 3
    @test tuned_model_lstm_0.model.range[1].upper == 10
    @test tuned_model_lstm_0.model.range[2].lower == 1
    @test tuned_model_lstm_0.model.range[2].upper == 6
    @test tuned_model_lstm_0.model.range[3].lower == 50
    @test tuned_model_lstm_0.model.range[3].upper == 500
    @test tuned_model_lstm_0.model.model.batch_size == 512
    @test tuned_model_lstm_0.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_1.model.range[1].lower == 3
    @test tuned_model_lstm_1.model.range[1].upper == 10
    @test tuned_model_lstm_1.model.range[2].lower == 1
    @test tuned_model_lstm_1.model.range[2].upper == 6
    @test tuned_model_lstm_1.model.range[3].lower == 50
    @test tuned_model_lstm_1.model.range[3].upper == 500
    @test tuned_model_lstm_1.model.model.batch_size == 512
    @test tuned_model_lstm_1.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_2.model.range[1].lower == 3
    @test tuned_model_lstm_2.model.range[1].upper == 10
    @test tuned_model_lstm_2.model.range[2].lower == 1
    @test tuned_model_lstm_2.model.range[2].upper == 6
    @test tuned_model_lstm_2.model.range[3].lower == 50
    @test tuned_model_lstm_2.model.range[3].upper == 500
    @test tuned_model_lstm_2.model.model.batch_size == 512
    @test tuned_model_lstm_2.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_3.model.range[1].lower == 3
    @test tuned_model_lstm_3.model.range[1].upper == 10
    @test tuned_model_lstm_3.model.range[2].lower == 1
    @test tuned_model_lstm_3.model.range[2].upper == 6
    @test tuned_model_lstm_3.model.range[3].lower == 50
    @test tuned_model_lstm_3.model.range[3].upper == 500
    @test tuned_model_lstm_3.model.model.batch_size == 512
    @test tuned_model_lstm_3.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_4.model.range[1].lower == 3
    @test tuned_model_lstm_4.model.range[1].upper == 10
    @test tuned_model_lstm_4.model.range[2].lower == 1
    @test tuned_model_lstm_4.model.range[2].upper == 6
    @test tuned_model_lstm_4.model.range[3].lower == 50
    @test tuned_model_lstm_4.model.range[3].upper == 500
    @test tuned_model_lstm_4.model.model.batch_size == 512
    @test tuned_model_lstm_4.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_5.model.range[1].lower == 3
    @test tuned_model_lstm_5.model.range[1].upper == 10
    @test tuned_model_lstm_5.model.range[2].lower == 1
    @test tuned_model_lstm_5.model.range[2].upper == 6
    @test tuned_model_lstm_5.model.range[3].lower == 50
    @test tuned_model_lstm_5.model.range[3].upper == 500
    @test tuned_model_lstm_5.model.model.batch_size == 512
    @test tuned_model_lstm_5.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_6.model.range[1].lower == 3
    @test tuned_model_lstm_6.model.range[1].upper == 10
    @test tuned_model_lstm_6.model.range[2].lower == 1
    @test tuned_model_lstm_6.model.range[2].upper == 6
    @test tuned_model_lstm_6.model.range[3].lower == 50
    @test tuned_model_lstm_6.model.range[3].upper == 500
    @test tuned_model_lstm_6.model.model.batch_size == 512
    @test tuned_model_lstm_6.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_7.model.range[1].lower == 3
    @test tuned_model_lstm_7.model.range[1].upper == 10
    @test tuned_model_lstm_7.model.range[2].lower == 1
    @test tuned_model_lstm_7.model.range[2].upper == 6
    @test tuned_model_lstm_7.model.range[3].lower == 50
    @test tuned_model_lstm_7.model.range[3].upper == 500
    @test tuned_model_lstm_7.model.model.batch_size == 512
    @test tuned_model_lstm_7.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_8.model.range[1].lower == 3
    @test tuned_model_lstm_8.model.range[1].upper == 10
    @test tuned_model_lstm_8.model.range[2].lower == 1
    @test tuned_model_lstm_8.model.range[2].upper == 6
    @test tuned_model_lstm_8.model.range[3].lower == 50
    @test tuned_model_lstm_8.model.range[3].upper == 500
    @test tuned_model_lstm_8.model.model.batch_size == 512
    @test tuned_model_lstm_8.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_9.model.range[1].lower == 3
    @test tuned_model_lstm_9.model.range[1].upper == 10
    @test tuned_model_lstm_9.model.range[2].lower == 1
    @test tuned_model_lstm_9.model.range[2].upper == 6
    @test tuned_model_lstm_9.model.range[3].lower == 50
    @test tuned_model_lstm_9.model.range[3].upper == 500
    @test tuned_model_lstm_9.model.model.batch_size == 512
    @test tuned_model_lstm_9.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_10.model.range[1].lower == 3
    @test tuned_model_lstm_10.model.range[1].upper == 10
    @test tuned_model_lstm_10.model.range[2].lower == 1
    @test tuned_model_lstm_10.model.range[2].upper == 6
    @test tuned_model_lstm_10.model.range[3].lower == 50
    @test tuned_model_lstm_10.model.range[3].upper == 500
    @test tuned_model_lstm_10.model.model.batch_size == 512
    @test tuned_model_lstm_10.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_11.model.range[1].lower == 3
    @test tuned_model_lstm_11.model.range[1].upper == 10
    @test tuned_model_lstm_11.model.range[2].lower == 1
    @test tuned_model_lstm_11.model.range[2].upper == 6
    @test tuned_model_lstm_11.model.range[3].lower == 50
    @test tuned_model_lstm_11.model.range[3].upper == 500
    @test tuned_model_lstm_11.model.model.batch_size == 512
    @test tuned_model_lstm_11.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_12.model.range[1].lower == 3
    @test tuned_model_lstm_12.model.range[1].upper == 10
    @test tuned_model_lstm_12.model.range[2].lower == 1
    @test tuned_model_lstm_12.model.range[2].upper == 6
    @test tuned_model_lstm_12.model.range[3].lower == 50
    @test tuned_model_lstm_12.model.range[3].upper == 500
    @test tuned_model_lstm_12.model.model.batch_size == 512
    @test tuned_model_lstm_12.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_13.model.range[1].lower == 3
    @test tuned_model_lstm_13.model.range[1].upper == 10
    @test tuned_model_lstm_13.model.range[2].lower == 1
    @test tuned_model_lstm_13.model.range[2].upper == 6
    @test tuned_model_lstm_13.model.range[3].lower == 50
    @test tuned_model_lstm_13.model.range[3].upper == 500
    @test tuned_model_lstm_13.model.model.batch_size == 512
    @test tuned_model_lstm_13.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_14.model.range[1].lower == 3
    @test tuned_model_lstm_14.model.range[1].upper == 10
    @test tuned_model_lstm_14.model.range[2].lower == 1
    @test tuned_model_lstm_14.model.range[2].upper == 6
    @test tuned_model_lstm_14.model.range[3].lower == 50
    @test tuned_model_lstm_14.model.range[3].upper == 500
    @test tuned_model_lstm_14.model.model.batch_size == 512
    @test tuned_model_lstm_14.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_15.model.range[1].lower == 3
    @test tuned_model_lstm_15.model.range[1].upper == 10
    @test tuned_model_lstm_15.model.range[2].lower == 1
    @test tuned_model_lstm_15.model.range[2].upper == 6
    @test tuned_model_lstm_15.model.range[3].lower == 50
    @test tuned_model_lstm_15.model.range[3].upper == 500
    @test tuned_model_lstm_15.model.model.batch_size == 512
    @test tuned_model_lstm_15.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_16.model.range[1].lower == 3
    @test tuned_model_lstm_16.model.range[1].upper == 10
    @test tuned_model_lstm_16.model.range[2].lower == 1
    @test tuned_model_lstm_16.model.range[2].upper == 6
    @test tuned_model_lstm_16.model.range[3].lower == 50
    @test tuned_model_lstm_16.model.range[3].upper == 500
    @test tuned_model_lstm_16.model.model.batch_size == 512
    @test tuned_model_lstm_16.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_17.model.range[1].lower == 3
    @test tuned_model_lstm_17.model.range[1].upper == 10
    @test tuned_model_lstm_17.model.range[2].lower == 1
    @test tuned_model_lstm_17.model.range[2].upper == 6
    @test tuned_model_lstm_17.model.range[3].lower == 50
    @test tuned_model_lstm_17.model.range[3].upper == 500
    @test tuned_model_lstm_17.model.model.batch_size == 512
    @test tuned_model_lstm_17.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_18.model.range[1].lower == 3
    @test tuned_model_lstm_18.model.range[1].upper == 10
    @test tuned_model_lstm_18.model.range[2].lower == 1
    @test tuned_model_lstm_18.model.range[2].upper == 6
    @test tuned_model_lstm_18.model.range[3].lower == 50
    @test tuned_model_lstm_18.model.range[3].upper == 500
    @test tuned_model_lstm_18.model.model.batch_size == 512
    @test tuned_model_lstm_18.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_19.model.range[1].lower == 3
    @test tuned_model_lstm_19.model.range[1].upper == 10
    @test tuned_model_lstm_19.model.range[2].lower == 1
    @test tuned_model_lstm_19.model.range[2].upper == 6
    @test tuned_model_lstm_19.model.range[3].lower == 50
    @test tuned_model_lstm_19.model.range[3].upper == 500
    @test tuned_model_lstm_19.model.model.batch_size == 512
    @test tuned_model_lstm_19.model.model.builder.σ == Flux.relu

    @test tuned_model_lstm_20.model.range[1].lower == 3
    @test tuned_model_lstm_20.model.range[1].upper == 10
    @test tuned_model_lstm_20.model.range[2].lower == 1
    @test tuned_model_lstm_20.model.range[2].upper == 6
    @test tuned_model_lstm_20.model.range[3].lower == 50
    @test tuned_model_lstm_20.model.range[3].upper == 500
    @test tuned_model_lstm_20.model.model.batch_size == 512
    @test tuned_model_lstm_20.model.model.builder.σ == Flux.relu

end


@testset "Neural Network Builder gru modification algorithms" begin

    architecture = "gru"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_gru_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_gru_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_gru_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_gru_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_gru_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_gru_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_gru_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    
    architecture = "gru"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_gru_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_gru_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_gru_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_gru_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_gru_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_gru_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_gru_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    architecture = "gru"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_gru_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_gru_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_gru_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_gru_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_gru_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_gru_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_gru_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            fraction_train = 0.8,
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_gru_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_gru_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_gru_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_gru_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_gru_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_gru_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_gru_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_gru_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_gru_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_gru_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_gru_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_gru_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_gru_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_gru_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_gru_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_gru_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_gru_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_gru_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_gru_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_gru_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_gru_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_gru_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_gru_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_gru_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_gru_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_gru_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_gru_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_gru_0.model.range[1].lower == 3
    @test tuned_model_gru_0.model.range[1].upper == 10
    @test tuned_model_gru_0.model.range[2].lower == 1
    @test tuned_model_gru_0.model.range[2].upper == 6
    @test tuned_model_gru_0.model.range[3].lower == 50
    @test tuned_model_gru_0.model.range[3].upper == 500
    @test tuned_model_gru_0.model.model.batch_size == 512
    @test tuned_model_gru_0.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_1.model.range[1].lower == 3
    @test tuned_model_gru_1.model.range[1].upper == 10
    @test tuned_model_gru_1.model.range[2].lower == 1
    @test tuned_model_gru_1.model.range[2].upper == 6
    @test tuned_model_gru_1.model.range[3].lower == 50
    @test tuned_model_gru_1.model.range[3].upper == 500
    @test tuned_model_gru_1.model.model.batch_size == 512
    @test tuned_model_gru_1.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_2.model.range[1].lower == 3
    @test tuned_model_gru_2.model.range[1].upper == 10
    @test tuned_model_gru_2.model.range[2].lower == 1
    @test tuned_model_gru_2.model.range[2].upper == 6
    @test tuned_model_gru_2.model.range[3].lower == 50
    @test tuned_model_gru_2.model.range[3].upper == 500
    @test tuned_model_gru_2.model.model.batch_size == 512
    @test tuned_model_gru_2.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_3.model.range[1].lower == 3
    @test tuned_model_gru_3.model.range[1].upper == 10
    @test tuned_model_gru_3.model.range[2].lower == 1
    @test tuned_model_gru_3.model.range[2].upper == 6
    @test tuned_model_gru_3.model.range[3].lower == 50
    @test tuned_model_gru_3.model.range[3].upper == 500
    @test tuned_model_gru_3.model.model.batch_size == 512
    @test tuned_model_gru_3.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_4.model.range[1].lower == 3
    @test tuned_model_gru_4.model.range[1].upper == 10
    @test tuned_model_gru_4.model.range[2].lower == 1
    @test tuned_model_gru_4.model.range[2].upper == 6
    @test tuned_model_gru_4.model.range[3].lower == 50
    @test tuned_model_gru_4.model.range[3].upper == 500
    @test tuned_model_gru_4.model.model.batch_size == 512
    @test tuned_model_gru_4.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_5.model.range[1].lower == 3
    @test tuned_model_gru_5.model.range[1].upper == 10
    @test tuned_model_gru_5.model.range[2].lower == 1
    @test tuned_model_gru_5.model.range[2].upper == 6
    @test tuned_model_gru_5.model.range[3].lower == 50
    @test tuned_model_gru_5.model.range[3].upper == 500
    @test tuned_model_gru_5.model.model.batch_size == 512
    @test tuned_model_gru_5.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_6.model.range[1].lower == 3
    @test tuned_model_gru_6.model.range[1].upper == 10
    @test tuned_model_gru_6.model.range[2].lower == 1
    @test tuned_model_gru_6.model.range[2].upper == 6
    @test tuned_model_gru_6.model.range[3].lower == 50
    @test tuned_model_gru_6.model.range[3].upper == 500
    @test tuned_model_gru_6.model.model.batch_size == 512
    @test tuned_model_gru_6.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_7.model.range[1].lower == 3
    @test tuned_model_gru_7.model.range[1].upper == 10
    @test tuned_model_gru_7.model.range[2].lower == 1
    @test tuned_model_gru_7.model.range[2].upper == 6
    @test tuned_model_gru_7.model.range[3].lower == 50
    @test tuned_model_gru_7.model.range[3].upper == 500
    @test tuned_model_gru_7.model.model.batch_size == 512
    @test tuned_model_gru_7.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_8.model.range[1].lower == 3
    @test tuned_model_gru_8.model.range[1].upper == 10
    @test tuned_model_gru_8.model.range[2].lower == 1
    @test tuned_model_gru_8.model.range[2].upper == 6
    @test tuned_model_gru_8.model.range[3].lower == 50
    @test tuned_model_gru_8.model.range[3].upper == 500
    @test tuned_model_gru_8.model.model.batch_size == 512
    @test tuned_model_gru_8.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_9.model.range[1].lower == 3
    @test tuned_model_gru_9.model.range[1].upper == 10
    @test tuned_model_gru_9.model.range[2].lower == 1
    @test tuned_model_gru_9.model.range[2].upper == 6
    @test tuned_model_gru_9.model.range[3].lower == 50
    @test tuned_model_gru_9.model.range[3].upper == 500
    @test tuned_model_gru_9.model.model.batch_size == 512
    @test tuned_model_gru_9.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_10.model.range[1].lower == 3
    @test tuned_model_gru_10.model.range[1].upper == 10
    @test tuned_model_gru_10.model.range[2].lower == 1
    @test tuned_model_gru_10.model.range[2].upper == 6
    @test tuned_model_gru_10.model.range[3].lower == 50
    @test tuned_model_gru_10.model.range[3].upper == 500
    @test tuned_model_gru_10.model.model.batch_size == 512
    @test tuned_model_gru_10.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_11.model.range[1].lower == 3
    @test tuned_model_gru_11.model.range[1].upper == 10
    @test tuned_model_gru_11.model.range[2].lower == 1
    @test tuned_model_gru_11.model.range[2].upper == 6
    @test tuned_model_gru_11.model.range[3].lower == 50
    @test tuned_model_gru_11.model.range[3].upper == 500
    @test tuned_model_gru_11.model.model.batch_size == 512
    @test tuned_model_gru_11.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_12.model.range[1].lower == 3
    @test tuned_model_gru_12.model.range[1].upper == 10
    @test tuned_model_gru_12.model.range[2].lower == 1
    @test tuned_model_gru_12.model.range[2].upper == 6
    @test tuned_model_gru_12.model.range[3].lower == 50
    @test tuned_model_gru_12.model.range[3].upper == 500
    @test tuned_model_gru_12.model.model.batch_size == 512
    @test tuned_model_gru_12.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_13.model.range[1].lower == 3
    @test tuned_model_gru_13.model.range[1].upper == 10
    @test tuned_model_gru_13.model.range[2].lower == 1
    @test tuned_model_gru_13.model.range[2].upper == 6
    @test tuned_model_gru_13.model.range[3].lower == 50
    @test tuned_model_gru_13.model.range[3].upper == 500
    @test tuned_model_gru_13.model.model.batch_size == 512
    @test tuned_model_gru_13.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_14.model.range[1].lower == 3
    @test tuned_model_gru_14.model.range[1].upper == 10
    @test tuned_model_gru_14.model.range[2].lower == 1
    @test tuned_model_gru_14.model.range[2].upper == 6
    @test tuned_model_gru_14.model.range[3].lower == 50
    @test tuned_model_gru_14.model.range[3].upper == 500
    @test tuned_model_gru_14.model.model.batch_size == 512
    @test tuned_model_gru_14.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_15.model.range[1].lower == 3
    @test tuned_model_gru_15.model.range[1].upper == 10
    @test tuned_model_gru_15.model.range[2].lower == 1
    @test tuned_model_gru_15.model.range[2].upper == 6
    @test tuned_model_gru_15.model.range[3].lower == 50
    @test tuned_model_gru_15.model.range[3].upper == 500
    @test tuned_model_gru_15.model.model.batch_size == 512
    @test tuned_model_gru_15.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_16.model.range[1].lower == 3
    @test tuned_model_gru_16.model.range[1].upper == 10
    @test tuned_model_gru_16.model.range[2].lower == 1
    @test tuned_model_gru_16.model.range[2].upper == 6
    @test tuned_model_gru_16.model.range[3].lower == 50
    @test tuned_model_gru_16.model.range[3].upper == 500
    @test tuned_model_gru_16.model.model.batch_size == 512
    @test tuned_model_gru_16.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_17.model.range[1].lower == 3
    @test tuned_model_gru_17.model.range[1].upper == 10
    @test tuned_model_gru_17.model.range[2].lower == 1
    @test tuned_model_gru_17.model.range[2].upper == 6
    @test tuned_model_gru_17.model.range[3].lower == 50
    @test tuned_model_gru_17.model.range[3].upper == 500
    @test tuned_model_gru_17.model.model.batch_size == 512
    @test tuned_model_gru_17.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_18.model.range[1].lower == 3
    @test tuned_model_gru_18.model.range[1].upper == 10
    @test tuned_model_gru_18.model.range[2].lower == 1
    @test tuned_model_gru_18.model.range[2].upper == 6
    @test tuned_model_gru_18.model.range[3].lower == 50
    @test tuned_model_gru_18.model.range[3].upper == 500
    @test tuned_model_gru_18.model.model.batch_size == 512
    @test tuned_model_gru_18.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_19.model.range[1].lower == 3
    @test tuned_model_gru_19.model.range[1].upper == 10
    @test tuned_model_gru_19.model.range[2].lower == 1
    @test tuned_model_gru_19.model.range[2].upper == 6
    @test tuned_model_gru_19.model.range[3].lower == 50
    @test tuned_model_gru_19.model.range[3].upper == 500
    @test tuned_model_gru_19.model.model.batch_size == 512
    @test tuned_model_gru_19.model.model.builder.σ == Flux.relu

    @test tuned_model_gru_20.model.range[1].lower == 3
    @test tuned_model_gru_20.model.range[1].upper == 10
    @test tuned_model_gru_20.model.range[2].lower == 1
    @test tuned_model_gru_20.model.range[2].upper == 6
    @test tuned_model_gru_20.model.range[3].lower == 50
    @test tuned_model_gru_20.model.range[3].upper == 500
    @test tuned_model_gru_20.model.model.batch_size == 512
    @test tuned_model_gru_20.model.model.builder.σ == Flux.relu

end

@testset "Neural Network Builder expn modification algorithms" begin

    architecture = "exploration_models"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_expn_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_expn_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_expn_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_expn_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_expn_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_expn_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_expn_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    
    processor = "cpu_threads"

    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    tuned_model_expn_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_expn_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_expn_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_expn_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_expn_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_expn_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_expn_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

   
    processor = "cpu_processes"

    algorithm = "adam"

    maximum_time = Dates.Minute(15)

    tuned_model_expn_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "radam"

    tuned_model_expn_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_expn_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_expn_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_expn_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_expn_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    algorithm = "pso"

    tuned_model_expn_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_expn_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_expn_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_expn_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_expn_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_expn_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_expn_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_expn_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_expn_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_expn_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_expn_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_expn_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_expn_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_expn_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_expn_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_expn_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_expn_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_expn_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_expn_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_expn_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_expn_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_expn_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_expn_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_expn_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_expn_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_expn_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_expn_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_expn_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_expn_0.model.range[1].lower == 3
    @test tuned_model_expn_0.model.range[1].upper == 10
    @test tuned_model_expn_0.model.range[2].lower == 1
    @test tuned_model_expn_0.model.range[2].upper == 6
    @test tuned_model_expn_0.model.range[3].lower == 50
    @test tuned_model_expn_0.model.range[3].upper == 500
    @test tuned_model_expn_0.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_0.model.model.batch_size == 512
    @test tuned_model_expn_0.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_1.model.range[1].lower == 3
    @test tuned_model_expn_1.model.range[1].upper == 10
    @test tuned_model_expn_1.model.range[2].lower == 1
    @test tuned_model_expn_1.model.range[2].upper == 6
    @test tuned_model_expn_1.model.range[3].lower == 50
    @test tuned_model_expn_1.model.range[3].upper == 500
    @test tuned_model_expn_1.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_1.model.model.batch_size == 512
    @test tuned_model_expn_1.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_2.model.range[1].lower == 3
    @test tuned_model_expn_2.model.range[1].upper == 10
    @test tuned_model_expn_2.model.range[2].lower == 1
    @test tuned_model_expn_2.model.range[2].upper == 6
    @test tuned_model_expn_2.model.range[3].lower == 50
    @test tuned_model_expn_2.model.range[3].upper == 500
    @test tuned_model_expn_2.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_2.model.model.batch_size == 512
    @test tuned_model_expn_2.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_3.model.range[1].lower == 3
    @test tuned_model_expn_3.model.range[1].upper == 10
    @test tuned_model_expn_3.model.range[2].lower == 1
    @test tuned_model_expn_3.model.range[2].upper == 6
    @test tuned_model_expn_3.model.range[3].lower == 50
    @test tuned_model_expn_3.model.range[3].upper == 500
    @test tuned_model_expn_3.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_3.model.model.batch_size == 512
    @test tuned_model_expn_3.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_4.model.range[1].lower == 3
    @test tuned_model_expn_4.model.range[1].upper == 10
    @test tuned_model_expn_4.model.range[2].lower == 1
    @test tuned_model_expn_4.model.range[2].upper == 6
    @test tuned_model_expn_4.model.range[3].lower == 50
    @test tuned_model_expn_4.model.range[3].upper == 500
    @test tuned_model_expn_4.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_4.model.model.batch_size == 512
    @test tuned_model_expn_4.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_5.model.range[1].lower == 3
    @test tuned_model_expn_5.model.range[1].upper == 10
    @test tuned_model_expn_5.model.range[2].lower == 1
    @test tuned_model_expn_5.model.range[2].upper == 6
    @test tuned_model_expn_5.model.range[3].lower == 50
    @test tuned_model_expn_5.model.range[3].upper == 500
    @test tuned_model_expn_5.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_5.model.model.batch_size == 512
    @test tuned_model_expn_5.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_6.model.range[1].lower == 3
    @test tuned_model_expn_6.model.range[1].upper == 10
    @test tuned_model_expn_6.model.range[2].lower == 1
    @test tuned_model_expn_6.model.range[2].upper == 6
    @test tuned_model_expn_6.model.range[3].lower == 50
    @test tuned_model_expn_6.model.range[3].upper == 500
    @test tuned_model_expn_6.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_6.model.model.batch_size == 512
    @test tuned_model_expn_6.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_7.model.range[1].lower == 3
    @test tuned_model_expn_7.model.range[1].upper == 10
    @test tuned_model_expn_7.model.range[2].lower == 1
    @test tuned_model_expn_7.model.range[2].upper == 6
    @test tuned_model_expn_7.model.range[3].lower == 50
    @test tuned_model_expn_7.model.range[3].upper == 500
    @test tuned_model_expn_7.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_7.model.model.batch_size == 512
    @test tuned_model_expn_7.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_8.model.range[1].lower == 3
    @test tuned_model_expn_8.model.range[1].upper == 10
    @test tuned_model_expn_8.model.range[2].lower == 1
    @test tuned_model_expn_8.model.range[2].upper == 6
    @test tuned_model_expn_8.model.range[3].lower == 50
    @test tuned_model_expn_8.model.range[3].upper == 500
    @test tuned_model_expn_8.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_8.model.model.batch_size == 512
    @test tuned_model_expn_8.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_9.model.range[1].lower == 3
    @test tuned_model_expn_9.model.range[1].upper == 10
    @test tuned_model_expn_9.model.range[2].lower == 1
    @test tuned_model_expn_9.model.range[2].upper == 6
    @test tuned_model_expn_9.model.range[3].lower == 50
    @test tuned_model_expn_9.model.range[3].upper == 500
    @test tuned_model_expn_9.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_9.model.model.batch_size == 512
    @test tuned_model_expn_9.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_10.model.range[1].lower == 3
    @test tuned_model_expn_10.model.range[1].upper == 10
    @test tuned_model_expn_10.model.range[2].lower == 1
    @test tuned_model_expn_10.model.range[2].upper == 6
    @test tuned_model_expn_10.model.range[3].lower == 50
    @test tuned_model_expn_10.model.range[3].upper == 500
    @test tuned_model_expn_10.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_10.model.model.batch_size == 512
    @test tuned_model_expn_10.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_11.model.range[1].lower == 3
    @test tuned_model_expn_11.model.range[1].upper == 10
    @test tuned_model_expn_11.model.range[2].lower == 1
    @test tuned_model_expn_11.model.range[2].upper == 6
    @test tuned_model_expn_11.model.range[3].lower == 50
    @test tuned_model_expn_11.model.range[3].upper == 500
    @test tuned_model_expn_11.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_11.model.model.batch_size == 512
    @test tuned_model_expn_11.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_12.model.range[1].lower == 3
    @test tuned_model_expn_12.model.range[1].upper == 10
    @test tuned_model_expn_12.model.range[2].lower == 1
    @test tuned_model_expn_12.model.range[2].upper == 6
    @test tuned_model_expn_12.model.range[3].lower == 50
    @test tuned_model_expn_12.model.range[3].upper == 500
    @test tuned_model_expn_12.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_12.model.model.batch_size == 512
    @test tuned_model_expn_12.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_13.model.range[1].lower == 3
    @test tuned_model_expn_13.model.range[1].upper == 10
    @test tuned_model_expn_13.model.range[2].lower == 1
    @test tuned_model_expn_13.model.range[2].upper == 6
    @test tuned_model_expn_13.model.range[3].lower == 50
    @test tuned_model_expn_13.model.range[3].upper == 500
    @test tuned_model_expn_13.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_13.model.model.batch_size == 512
    @test tuned_model_expn_13.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_14.model.range[1].lower == 3
    @test tuned_model_expn_14.model.range[1].upper == 10
    @test tuned_model_expn_14.model.range[2].lower == 1
    @test tuned_model_expn_14.model.range[2].upper == 6
    @test tuned_model_expn_14.model.range[3].lower == 50
    @test tuned_model_expn_14.model.range[3].upper == 500
    @test tuned_model_expn_14.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_14.model.model.batch_size == 512
    @test tuned_model_expn_14.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_15.model.range[1].lower == 3
    @test tuned_model_expn_15.model.range[1].upper == 10
    @test tuned_model_expn_15.model.range[2].lower == 1
    @test tuned_model_expn_15.model.range[2].upper == 6
    @test tuned_model_expn_15.model.range[3].lower == 50
    @test tuned_model_expn_15.model.range[3].upper == 500
    @test tuned_model_expn_15.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_15.model.model.batch_size == 512
    @test tuned_model_expn_15.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_16.model.range[1].lower == 3
    @test tuned_model_expn_16.model.range[1].upper == 10
    @test tuned_model_expn_16.model.range[2].lower == 1
    @test tuned_model_expn_16.model.range[2].upper == 6
    @test tuned_model_expn_16.model.range[3].lower == 50
    @test tuned_model_expn_16.model.range[3].upper == 500
    @test tuned_model_expn_16.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_16.model.model.batch_size == 512
    @test tuned_model_expn_16.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_17.model.range[1].lower == 3
    @test tuned_model_expn_17.model.range[1].upper == 10
    @test tuned_model_expn_17.model.range[2].lower == 1
    @test tuned_model_expn_17.model.range[2].upper == 6
    @test tuned_model_expn_17.model.range[3].lower == 50
    @test tuned_model_expn_17.model.range[3].upper == 500
    @test tuned_model_expn_17.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_17.model.model.batch_size == 512
    @test tuned_model_expn_17.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_18.model.range[1].lower == 3
    @test tuned_model_expn_18.model.range[1].upper == 10
    @test tuned_model_expn_18.model.range[2].lower == 1
    @test tuned_model_expn_18.model.range[2].upper == 6
    @test tuned_model_expn_18.model.range[3].lower == 50
    @test tuned_model_expn_18.model.range[3].upper == 500
    @test tuned_model_expn_18.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_18.model.model.batch_size == 512
    @test tuned_model_expn_18.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_19.model.range[1].lower == 3
    @test tuned_model_expn_19.model.range[1].upper == 10
    @test tuned_model_expn_19.model.range[2].lower == 1
    @test tuned_model_expn_19.model.range[2].upper == 6
    @test tuned_model_expn_19.model.range[3].lower == 50
    @test tuned_model_expn_19.model.range[3].upper == 500
    @test tuned_model_expn_19.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_19.model.model.batch_size == 512
    @test tuned_model_expn_19.model.model.builder.σ == Flux.relu

    @test tuned_model_expn_20.model.range[1].lower == 3
    @test tuned_model_expn_20.model.range[1].upper == 10
    @test tuned_model_expn_20.model.range[2].lower == 1
    @test tuned_model_expn_20.model.range[2].upper == 6
    @test tuned_model_expn_20.model.range[3].lower == 50
    @test tuned_model_expn_20.model.range[3].upper == 500
    @test tuned_model_expn_20.model.range[4].values == ("Fnn", "Rbf", "Icnn", "ResNet", "PolyNet", "DenseNet",)
    @test tuned_model_expn_20.model.model.batch_size == 512
    @test tuned_model_expn_20.model.model.builder.σ == Flux.relu

end




###############################
### Parameters modification ###
###############################



###############################
### Parameters modification ###
###############################



###############################
### Parameters modification ###
###############################




###############################
### Parameters modification ###
###############################

@testset "Neural Network Builder Fnn modification algorithms" begin

    architecture = "fnn"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_fnn_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_fnn_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_fnn_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_fnn_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_fnn_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_fnn_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_fnn_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "fnn"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_fnn_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_fnn_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_fnn_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_fnn_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_fnn_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_fnn_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_fnn_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "fnn"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_fnn_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_fnn_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_fnn_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_fnn_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_fnn_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_fnn_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_fnn_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

        

    # Algorithm verification
    @test typeof(tuned_model_fnn_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_fnn_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_fnn_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_fnn_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_fnn_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_fnn_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_fnn_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_fnn_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_fnn_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_fnn_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_fnn_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_fnn_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_fnn_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_fnn_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_fnn_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_fnn_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_fnn_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_fnn_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_fnn_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_fnn_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_fnn_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_fnn_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_fnn_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_fnn_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_fnn_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_fnn_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_fnn_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_fnn_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_fnn_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_fnn_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_fnn_0.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_0.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_0.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_0.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_0.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_0.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_0.model.model.batch_size == batch_size
    @test tuned_model_fnn_0.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_1.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_1.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_1.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_1.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_1.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_1.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_1.model.model.batch_size == batch_size
    @test tuned_model_fnn_1.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_2.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_2.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_2.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_2.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_2.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_2.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_2.model.model.batch_size == batch_size
    @test tuned_model_fnn_2.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_3.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_3.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_3.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_3.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_3.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_3.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_3.model.model.batch_size == batch_size
    @test tuned_model_fnn_3.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_4.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_4.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_4.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_4.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_4.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_4.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_4.model.model.batch_size == batch_size
    @test tuned_model_fnn_4.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_5.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_5.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_5.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_5.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_5.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_5.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_5.model.model.batch_size == batch_size
    @test tuned_model_fnn_5.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_6.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_6.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_6.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_6.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_6.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_6.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_6.model.model.batch_size == batch_size
    @test tuned_model_fnn_6.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_7.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_7.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_7.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_7.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_7.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_7.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_7.model.model.batch_size == batch_size
    @test tuned_model_fnn_7.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_8.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_8.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_8.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_8.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_8.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_8.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_8.model.model.batch_size == batch_size
    @test tuned_model_fnn_8.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_9.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_9.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_9.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_9.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_9.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_9.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_9.model.model.batch_size == batch_size
    @test tuned_model_fnn_9.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_10.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_10.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_10.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_10.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_10.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_10.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_10.model.model.batch_size == batch_size
    @test tuned_model_fnn_10.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_11.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_11.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_11.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_11.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_11.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_11.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_11.model.model.batch_size == batch_size
    @test tuned_model_fnn_11.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_12.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_12.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_12.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_12.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_12.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_12.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_12.model.model.batch_size == batch_size
    @test tuned_model_fnn_12.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_13.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_13.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_13.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_13.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_13.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_13.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_13.model.model.batch_size == batch_size
    @test tuned_model_fnn_13.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_14.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_14.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_14.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_14.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_14.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_14.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_14.model.model.batch_size == batch_size
    @test tuned_model_fnn_14.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_15.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_15.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_15.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_15.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_15.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_15.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_15.model.model.batch_size == batch_size
    @test tuned_model_fnn_15.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_16.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_16.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_16.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_16.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_16.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_16.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_16.model.model.batch_size == batch_size
    @test tuned_model_fnn_16.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_17.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_17.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_17.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_17.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_17.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_17.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_17.model.model.batch_size == batch_size
    @test tuned_model_fnn_17.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_18.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_18.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_18.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_18.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_18.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_18.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_18.model.model.batch_size == batch_size
    @test tuned_model_fnn_18.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_19.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_19.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_19.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_19.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_19.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_19.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_19.model.model.batch_size == batch_size
    @test tuned_model_fnn_19.model.model.builder.σ == Flux.swish

    @test tuned_model_fnn_20.model.range[1].lower == minimum_neuron
    @test tuned_model_fnn_20.model.range[1].upper == maximum_neuron
    @test tuned_model_fnn_20.model.range[2].lower == minimum_layers
    @test tuned_model_fnn_20.model.range[2].upper == maximum_layers
    @test tuned_model_fnn_20.model.range[3].lower == minimum_epochs
    @test tuned_model_fnn_20.model.range[3].upper == maximum_epochs
    @test tuned_model_fnn_20.model.model.batch_size == batch_size
    @test tuned_model_fnn_20.model.model.builder.σ == Flux.swish

end

@testset "Neural Network Builder icnn modification algorithms" begin

    architecture = "icnn"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_icnn_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_icnn_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_icnn_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_icnn_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_icnn_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_icnn_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_icnn_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "icnn"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_icnn_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_icnn_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_icnn_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_icnn_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_icnn_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_icnn_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_icnn_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "icnn"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_icnn_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_icnn_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_icnn_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_icnn_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_icnn_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_icnn_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_icnn_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

        

    # Algorithm verification
    @test typeof(tuned_model_icnn_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_icnn_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_icnn_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_icnn_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_icnn_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_icnn_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_icnn_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_icnn_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_icnn_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_icnn_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_icnn_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_icnn_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_icnn_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_icnn_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_icnn_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_icnn_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_icnn_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_icnn_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_icnn_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_icnn_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_icnn_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_icnn_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_icnn_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_icnn_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_icnn_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_icnn_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_icnn_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_icnn_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_icnn_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_icnn_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_icnn_0.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_0.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_0.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_0.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_0.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_0.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_0.model.model.batch_size == batch_size
    @test tuned_model_icnn_0.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_1.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_1.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_1.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_1.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_1.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_1.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_1.model.model.batch_size == batch_size
    @test tuned_model_icnn_1.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_2.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_2.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_2.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_2.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_2.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_2.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_2.model.model.batch_size == batch_size
    @test tuned_model_icnn_2.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_3.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_3.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_3.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_3.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_3.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_3.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_3.model.model.batch_size == batch_size
    @test tuned_model_icnn_3.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_4.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_4.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_4.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_4.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_4.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_4.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_4.model.model.batch_size == batch_size
    @test tuned_model_icnn_4.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_5.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_5.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_5.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_5.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_5.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_5.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_5.model.model.batch_size == batch_size
    @test tuned_model_icnn_5.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_6.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_6.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_6.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_6.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_6.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_6.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_6.model.model.batch_size == batch_size
    @test tuned_model_icnn_6.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_7.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_7.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_7.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_7.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_7.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_7.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_7.model.model.batch_size == batch_size
    @test tuned_model_icnn_7.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_8.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_8.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_8.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_8.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_8.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_8.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_8.model.model.batch_size == batch_size
    @test tuned_model_icnn_8.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_9.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_9.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_9.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_9.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_9.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_9.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_9.model.model.batch_size == batch_size
    @test tuned_model_icnn_9.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_10.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_10.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_10.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_10.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_10.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_10.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_10.model.model.batch_size == batch_size
    @test tuned_model_icnn_10.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_11.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_11.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_11.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_11.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_11.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_11.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_11.model.model.batch_size == batch_size
    @test tuned_model_icnn_11.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_12.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_12.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_12.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_12.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_12.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_12.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_12.model.model.batch_size == batch_size
    @test tuned_model_icnn_12.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_13.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_13.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_13.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_13.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_13.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_13.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_13.model.model.batch_size == batch_size
    @test tuned_model_icnn_13.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_14.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_14.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_14.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_14.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_14.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_14.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_14.model.model.batch_size == batch_size
    @test tuned_model_icnn_14.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_15.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_15.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_15.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_15.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_15.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_15.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_15.model.model.batch_size == batch_size
    @test tuned_model_icnn_15.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_16.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_16.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_16.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_16.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_16.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_16.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_16.model.model.batch_size == batch_size
    @test tuned_model_icnn_16.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_17.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_17.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_17.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_17.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_17.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_17.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_17.model.model.batch_size == batch_size
    @test tuned_model_icnn_17.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_18.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_18.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_18.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_18.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_18.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_18.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_18.model.model.batch_size == batch_size
    @test tuned_model_icnn_18.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_19.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_19.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_19.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_19.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_19.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_19.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_19.model.model.batch_size == batch_size
    @test tuned_model_icnn_19.model.model.builder.σ == Flux.relu

    @test tuned_model_icnn_20.model.range[1].lower == minimum_neuron
    @test tuned_model_icnn_20.model.range[1].upper == maximum_neuron
    @test tuned_model_icnn_20.model.range[2].lower == minimum_layers
    @test tuned_model_icnn_20.model.range[2].upper == maximum_layers
    @test tuned_model_icnn_20.model.range[3].lower == minimum_epochs
    @test tuned_model_icnn_20.model.range[3].upper == maximum_epochs
    @test tuned_model_icnn_20.model.model.batch_size == batch_size
    @test tuned_model_icnn_20.model.model.builder.σ == Flux.relu

end

@testset "Neural Network Builder resnet modification algorithms" begin

    architecture = "resnet"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_resnet_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_resnet_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_resnet_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_resnet_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_resnet_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_resnet_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_resnet_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "resnet"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_resnet_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_resnet_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_resnet_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_resnet_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_resnet_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_resnet_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_resnet_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "resnet"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_resnet_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_resnet_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_resnet_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_resnet_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_resnet_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_resnet_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_resnet_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

        

    # Algorithm verification
    @test typeof(tuned_model_resnet_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_resnet_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_resnet_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_resnet_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_resnet_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_resnet_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_resnet_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_resnet_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_resnet_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_resnet_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_resnet_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_resnet_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_resnet_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_resnet_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_resnet_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_resnet_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_resnet_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_resnet_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_resnet_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_resnet_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_resnet_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_resnet_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_resnet_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_resnet_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_resnet_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_resnet_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_resnet_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_resnet_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_resnet_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_resnet_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_resnet_0.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_0.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_0.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_0.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_0.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_0.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_0.model.model.batch_size == batch_size
    @test tuned_model_resnet_0.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_1.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_1.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_1.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_1.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_1.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_1.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_1.model.model.batch_size == batch_size
    @test tuned_model_resnet_1.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_2.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_2.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_2.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_2.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_2.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_2.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_2.model.model.batch_size == batch_size
    @test tuned_model_resnet_2.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_3.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_3.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_3.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_3.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_3.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_3.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_3.model.model.batch_size == batch_size
    @test tuned_model_resnet_3.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_4.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_4.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_4.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_4.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_4.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_4.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_4.model.model.batch_size == batch_size
    @test tuned_model_resnet_4.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_5.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_5.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_5.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_5.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_5.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_5.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_5.model.model.batch_size == batch_size
    @test tuned_model_resnet_5.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_6.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_6.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_6.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_6.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_6.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_6.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_6.model.model.batch_size == batch_size
    @test tuned_model_resnet_6.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_7.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_7.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_7.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_7.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_7.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_7.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_7.model.model.batch_size == batch_size
    @test tuned_model_resnet_7.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_8.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_8.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_8.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_8.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_8.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_8.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_8.model.model.batch_size == batch_size
    @test tuned_model_resnet_8.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_9.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_9.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_9.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_9.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_9.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_9.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_9.model.model.batch_size == batch_size
    @test tuned_model_resnet_9.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_10.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_10.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_10.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_10.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_10.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_10.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_10.model.model.batch_size == batch_size
    @test tuned_model_resnet_10.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_11.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_11.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_11.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_11.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_11.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_11.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_11.model.model.batch_size == batch_size
    @test tuned_model_resnet_11.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_12.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_12.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_12.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_12.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_12.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_12.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_12.model.model.batch_size == batch_size
    @test tuned_model_resnet_12.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_13.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_13.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_13.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_13.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_13.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_13.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_13.model.model.batch_size == batch_size
    @test tuned_model_resnet_13.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_14.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_14.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_14.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_14.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_14.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_14.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_14.model.model.batch_size == batch_size
    @test tuned_model_resnet_14.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_15.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_15.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_15.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_15.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_15.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_15.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_15.model.model.batch_size == batch_size
    @test tuned_model_resnet_15.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_16.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_16.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_16.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_16.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_16.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_16.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_16.model.model.batch_size == batch_size
    @test tuned_model_resnet_16.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_17.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_17.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_17.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_17.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_17.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_17.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_17.model.model.batch_size == batch_size
    @test tuned_model_resnet_17.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_18.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_18.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_18.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_18.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_18.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_18.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_18.model.model.batch_size == batch_size
    @test tuned_model_resnet_18.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_19.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_19.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_19.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_19.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_19.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_19.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_19.model.model.batch_size == batch_size
    @test tuned_model_resnet_19.model.model.builder.σ == Flux.swish

    @test tuned_model_resnet_20.model.range[1].lower == minimum_neuron
    @test tuned_model_resnet_20.model.range[1].upper == maximum_neuron
    @test tuned_model_resnet_20.model.range[2].lower == minimum_layers
    @test tuned_model_resnet_20.model.range[2].upper == maximum_layers
    @test tuned_model_resnet_20.model.range[3].lower == minimum_epochs
    @test tuned_model_resnet_20.model.range[3].upper == maximum_epochs
    @test tuned_model_resnet_20.model.model.batch_size == batch_size
    @test tuned_model_resnet_20.model.model.builder.σ == Flux.swish

end

@testset "Neural Network Builder densenet modification algorithms" begin

    architecture = "densenet"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_densenet_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_densenet_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_densenet_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_densenet_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_densenet_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_densenet_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_densenet_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "densenet"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_densenet_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_densenet_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_densenet_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_densenet_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_densenet_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_densenet_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_densenet_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "densenet"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_densenet_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_densenet_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_densenet_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_densenet_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_densenet_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_densenet_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_densenet_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

        

    # Algorithm verification
    @test typeof(tuned_model_densenet_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_densenet_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_densenet_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_densenet_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_densenet_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_densenet_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_densenet_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_densenet_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_densenet_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_densenet_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_densenet_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_densenet_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_densenet_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_densenet_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_densenet_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_densenet_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_densenet_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_densenet_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_densenet_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_densenet_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_densenet_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_densenet_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_densenet_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_densenet_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_densenet_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_densenet_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_densenet_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_densenet_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_densenet_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_densenet_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_densenet_0.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_0.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_0.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_0.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_0.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_0.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_0.model.model.batch_size == batch_size
    @test tuned_model_densenet_0.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_1.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_1.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_1.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_1.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_1.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_1.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_1.model.model.batch_size == batch_size
    @test tuned_model_densenet_1.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_2.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_2.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_2.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_2.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_2.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_2.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_2.model.model.batch_size == batch_size
    @test tuned_model_densenet_2.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_3.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_3.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_3.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_3.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_3.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_3.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_3.model.model.batch_size == batch_size
    @test tuned_model_densenet_3.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_4.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_4.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_4.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_4.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_4.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_4.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_4.model.model.batch_size == batch_size
    @test tuned_model_densenet_4.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_5.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_5.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_5.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_5.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_5.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_5.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_5.model.model.batch_size == batch_size
    @test tuned_model_densenet_5.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_6.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_6.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_6.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_6.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_6.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_6.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_6.model.model.batch_size == batch_size
    @test tuned_model_densenet_6.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_7.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_7.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_7.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_7.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_7.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_7.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_7.model.model.batch_size == batch_size
    @test tuned_model_densenet_7.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_8.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_8.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_8.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_8.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_8.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_8.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_8.model.model.batch_size == batch_size
    @test tuned_model_densenet_8.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_9.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_9.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_9.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_9.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_9.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_9.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_9.model.model.batch_size == batch_size
    @test tuned_model_densenet_9.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_10.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_10.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_10.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_10.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_10.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_10.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_10.model.model.batch_size == batch_size
    @test tuned_model_densenet_10.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_11.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_11.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_11.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_11.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_11.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_11.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_11.model.model.batch_size == batch_size
    @test tuned_model_densenet_11.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_12.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_12.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_12.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_12.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_12.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_12.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_12.model.model.batch_size == batch_size
    @test tuned_model_densenet_12.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_13.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_13.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_13.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_13.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_13.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_13.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_13.model.model.batch_size == batch_size
    @test tuned_model_densenet_13.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_14.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_14.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_14.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_14.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_14.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_14.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_14.model.model.batch_size == batch_size
    @test tuned_model_densenet_14.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_15.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_15.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_15.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_15.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_15.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_15.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_15.model.model.batch_size == batch_size
    @test tuned_model_densenet_15.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_16.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_16.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_16.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_16.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_16.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_16.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_16.model.model.batch_size == batch_size
    @test tuned_model_densenet_16.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_17.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_17.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_17.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_17.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_17.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_17.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_17.model.model.batch_size == batch_size
    @test tuned_model_densenet_17.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_18.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_18.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_18.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_18.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_18.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_18.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_18.model.model.batch_size == batch_size
    @test tuned_model_densenet_18.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_19.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_19.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_19.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_19.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_19.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_19.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_19.model.model.batch_size == batch_size
    @test tuned_model_densenet_19.model.model.builder.σ == Flux.swish

    @test tuned_model_densenet_20.model.range[1].lower == minimum_neuron
    @test tuned_model_densenet_20.model.range[1].upper == maximum_neuron
    @test tuned_model_densenet_20.model.range[2].lower == minimum_layers
    @test tuned_model_densenet_20.model.range[2].upper == maximum_layers
    @test tuned_model_densenet_20.model.range[3].lower == minimum_epochs
    @test tuned_model_densenet_20.model.range[3].upper == maximum_epochs
    @test tuned_model_densenet_20.model.model.batch_size == batch_size
    @test tuned_model_densenet_20.model.model.builder.σ == Flux.swish

end

@testset "Neural Network Builder polynet modification algorithms" begin

    architecture = "polynet"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_polynet_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_polynet_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_polynet_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_polynet_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_polynet_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_polynet_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_polynet_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "polynet"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_polynet_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_polynet_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_polynet_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_polynet_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_polynet_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_polynet_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_polynet_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "polynet"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_polynet_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_polynet_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_polynet_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_polynet_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_polynet_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_polynet_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_polynet_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

        

    # Algorithm verification
    @test typeof(tuned_model_polynet_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_polynet_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_polynet_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_polynet_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_polynet_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_polynet_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_polynet_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_polynet_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_polynet_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_polynet_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_polynet_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_polynet_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_polynet_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_polynet_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_polynet_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_polynet_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_polynet_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_polynet_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_polynet_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_polynet_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_polynet_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_polynet_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_polynet_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_polynet_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_polynet_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_polynet_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_polynet_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_polynet_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_polynet_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_polynet_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_polynet_0.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_0.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_0.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_0.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_0.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_0.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_0.model.model.batch_size == batch_size
    @test tuned_model_polynet_0.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_1.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_1.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_1.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_1.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_1.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_1.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_1.model.model.batch_size == batch_size
    @test tuned_model_polynet_1.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_2.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_2.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_2.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_2.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_2.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_2.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_2.model.model.batch_size == batch_size
    @test tuned_model_polynet_2.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_3.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_3.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_3.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_3.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_3.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_3.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_3.model.model.batch_size == batch_size
    @test tuned_model_polynet_3.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_4.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_4.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_4.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_4.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_4.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_4.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_4.model.model.batch_size == batch_size
    @test tuned_model_polynet_4.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_5.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_5.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_5.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_5.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_5.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_5.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_5.model.model.batch_size == batch_size
    @test tuned_model_polynet_5.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_6.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_6.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_6.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_6.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_6.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_6.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_6.model.model.batch_size == batch_size
    @test tuned_model_polynet_6.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_7.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_7.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_7.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_7.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_7.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_7.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_7.model.model.batch_size == batch_size
    @test tuned_model_polynet_7.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_8.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_8.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_8.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_8.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_8.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_8.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_8.model.model.batch_size == batch_size
    @test tuned_model_polynet_8.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_9.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_9.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_9.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_9.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_9.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_9.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_9.model.model.batch_size == batch_size
    @test tuned_model_polynet_9.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_10.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_10.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_10.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_10.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_10.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_10.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_10.model.model.batch_size == batch_size
    @test tuned_model_polynet_10.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_11.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_11.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_11.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_11.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_11.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_11.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_11.model.model.batch_size == batch_size
    @test tuned_model_polynet_11.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_12.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_12.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_12.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_12.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_12.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_12.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_12.model.model.batch_size == batch_size
    @test tuned_model_polynet_12.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_13.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_13.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_13.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_13.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_13.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_13.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_13.model.model.batch_size == batch_size
    @test tuned_model_polynet_13.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_14.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_14.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_14.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_14.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_14.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_14.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_14.model.model.batch_size == batch_size
    @test tuned_model_polynet_14.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_15.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_15.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_15.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_15.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_15.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_15.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_15.model.model.batch_size == batch_size
    @test tuned_model_polynet_15.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_16.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_16.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_16.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_16.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_16.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_16.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_16.model.model.batch_size == batch_size
    @test tuned_model_polynet_16.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_17.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_17.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_17.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_17.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_17.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_17.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_17.model.model.batch_size == batch_size
    @test tuned_model_polynet_17.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_18.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_18.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_18.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_18.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_18.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_18.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_18.model.model.batch_size == batch_size
    @test tuned_model_polynet_18.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_19.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_19.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_19.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_19.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_19.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_19.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_19.model.model.batch_size == batch_size
    @test tuned_model_polynet_19.model.model.builder.σ == Flux.swish

    @test tuned_model_polynet_20.model.range[1].lower == minimum_neuron
    @test tuned_model_polynet_20.model.range[1].upper == maximum_neuron
    @test tuned_model_polynet_20.model.range[2].lower == minimum_layers
    @test tuned_model_polynet_20.model.range[2].upper == maximum_layers
    @test tuned_model_polynet_20.model.range[3].lower == minimum_epochs
    @test tuned_model_polynet_20.model.range[3].upper == maximum_epochs
    @test tuned_model_polynet_20.model.model.batch_size == batch_size
    @test tuned_model_polynet_20.model.model.builder.σ == Flux.swish

end

@testset "Neural Network Builder neuralnetODE type 1 modification algorithms" begin

    architecture = "neuralnet_ode_type1"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_neuralnetODE_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "neuralnet_ode_type1"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_neuralnetODE_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "neuralnet_ode_type1"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_neuralnetODE_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

        

    # Algorithm verification
    @test typeof(tuned_model_neuralnetODE_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_neuralnetODE_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_neuralnetODE_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_neuralnetODE_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_neuralnetODE_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_neuralnetODE_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_neuralnetODE_0.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_0.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_0.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_0.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_0.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_0.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_0.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_0.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_1.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_1.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_1.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_1.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_1.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_1.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_1.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_1.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_2.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_2.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_2.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_2.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_2.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_2.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_2.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_2.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_3.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_3.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_3.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_3.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_3.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_3.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_3.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_3.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_4.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_4.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_4.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_4.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_4.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_4.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_4.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_4.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_5.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_5.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_5.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_5.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_5.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_5.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_5.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_5.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_6.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_6.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_6.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_6.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_6.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_6.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_6.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_6.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_7.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_7.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_7.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_7.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_7.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_7.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_7.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_7.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_8.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_8.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_8.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_8.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_8.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_8.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_8.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_8.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_9.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_9.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_9.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_9.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_9.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_9.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_9.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_9.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_10.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_10.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_10.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_10.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_10.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_10.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_10.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_10.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_11.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_11.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_11.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_11.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_11.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_11.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_11.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_11.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_12.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_12.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_12.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_12.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_12.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_12.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_12.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_12.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_13.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_13.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_13.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_13.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_13.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_13.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_13.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_13.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_14.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_14.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_14.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_14.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_14.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_14.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_14.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_14.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_15.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_15.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_15.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_15.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_15.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_15.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_15.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_15.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_16.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_16.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_16.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_16.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_16.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_16.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_16.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_16.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_17.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_17.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_17.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_17.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_17.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_17.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_17.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_17.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_18.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_18.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_18.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_18.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_18.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_18.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_18.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_18.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_19.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_19.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_19.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_19.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_19.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_19.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_19.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_19.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_20.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_20.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_20.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_20.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_20.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_20.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_20.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_20.model.model.builder.σ == Flux.swish

end

@testset "Neural Network Builder neuralnetODE type 2 modification algorithms" begin

    architecture = "neuralnet_ode_type2"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_neuralnetODE_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "neuralnet_ode_type2"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_neuralnetODE_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    architecture = "neuralnet_ode_type2"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_neuralnetODE_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,

        ) #dispatched function

    algorithm = "radam"

    tuned_model_neuralnetODE_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_neuralnetODE_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_neuralnetODE_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_neuralnetODE_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_neuralnetODE_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_neuralnetODE_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

        

    # Algorithm verification
    @test typeof(tuned_model_neuralnetODE_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_neuralnetODE_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_neuralnetODE_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_neuralnetODE_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_neuralnetODE_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_neuralnetODE_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_neuralnetODE_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_neuralnetODE_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_neuralnetODE_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_neuralnetODE_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_neuralnetODE_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_neuralnetODE_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_neuralnetODE_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_neuralnetODE_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_neuralnetODE_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_neuralnetODE_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_neuralnetODE_0.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_0.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_0.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_0.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_0.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_0.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_0.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_0.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_1.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_1.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_1.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_1.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_1.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_1.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_1.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_1.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_2.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_2.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_2.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_2.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_2.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_2.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_2.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_2.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_3.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_3.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_3.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_3.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_3.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_3.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_3.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_3.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_4.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_4.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_4.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_4.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_4.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_4.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_4.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_4.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_5.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_5.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_5.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_5.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_5.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_5.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_5.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_5.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_6.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_6.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_6.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_6.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_6.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_6.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_6.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_6.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_7.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_7.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_7.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_7.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_7.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_7.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_7.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_7.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_8.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_8.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_8.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_8.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_8.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_8.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_8.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_8.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_9.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_9.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_9.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_9.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_9.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_9.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_9.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_9.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_10.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_10.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_10.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_10.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_10.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_10.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_10.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_10.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_11.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_11.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_11.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_11.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_11.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_11.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_11.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_11.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_12.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_12.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_12.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_12.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_12.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_12.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_12.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_12.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_13.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_13.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_13.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_13.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_13.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_13.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_13.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_13.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_14.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_14.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_14.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_14.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_14.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_14.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_14.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_14.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_15.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_15.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_15.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_15.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_15.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_15.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_15.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_15.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_16.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_16.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_16.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_16.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_16.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_16.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_16.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_16.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_17.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_17.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_17.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_17.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_17.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_17.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_17.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_17.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_18.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_18.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_18.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_18.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_18.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_18.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_18.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_18.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_19.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_19.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_19.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_19.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_19.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_19.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_19.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_19.model.model.builder.σ == Flux.swish

    @test tuned_model_neuralnetODE_20.model.range[1].lower == minimum_neuron
    @test tuned_model_neuralnetODE_20.model.range[1].upper == maximum_neuron
    @test tuned_model_neuralnetODE_20.model.range[2].lower == minimum_layers
    @test tuned_model_neuralnetODE_20.model.range[2].upper == maximum_layers
    @test tuned_model_neuralnetODE_20.model.range[3].lower == minimum_epochs
    @test tuned_model_neuralnetODE_20.model.range[3].upper == maximum_epochs
    @test tuned_model_neuralnetODE_20.model.model.batch_size == batch_size
    @test tuned_model_neuralnetODE_20.model.model.builder.σ == Flux.swish

end

@testset "Neural Network Builder rbf modification algorithms" begin

    architecture = "rbf"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_rbf_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_rbf_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_rbf_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_rbf_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rbf_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rbf_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_rbf_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
        ) #dispatched function

    processor = "cpu_threads"

    algorithm = "adam"

    tuned_model_rbf_7 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "radam"

    tuned_model_rbf_8 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "nadam"

    tuned_model_rbf_9 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "oadam"

    tuned_model_rbf_10 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rbf_11 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rbf_12 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "pso"

    tuned_model_rbf_13 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function


    processor = "cpu_processes"

    algorithm = "adam"


    tuned_model_rbf_14 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "radam"

    tuned_model_rbf_15 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "nadam"

    tuned_model_rbf_16 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "oadam"

    tuned_model_rbf_17 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rbf_18 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rbf_19 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    algorithm = "pso"

    tuned_model_rbf_20 = _neural_network_builder(
        ARCHITECTURE_LIST[Symbol(architecture)],
        PROCESSOR_LIST[Symbol(processor)],
        ALGORITHM_LIST[Symbol(algorithm)],
        maximum_time;
        neuralnet_minimum_epochs = minimum_epochs,
        neuralnet_maximum_epochs = maximum_epochs,
        neuralnet_minimum_neuron = minimum_neuron,
        neuralnet_maximum_neuron = maximum_neuron,
        neuralnet_batch_size = batch_size,
    ) #dispatched function

    
    # Algorithm verification
    @test typeof(tuned_model_rbf_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rbf_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rbf_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rbf_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rbf_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rbf_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rbf_6.model.model.optimiser) == Optim.ParticleSwarm{Any}
 
    @test typeof(tuned_model_rbf_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rbf_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rbf_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rbf_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rbf_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rbf_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rbf_13.model.model.optimiser) == Optim.ParticleSwarm{Any}
 
    @test typeof(tuned_model_rbf_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rbf_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rbf_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rbf_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rbf_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rbf_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rbf_20.model.model.optimiser) == Optim.ParticleSwarm{Any}
 
    # Processors verification
    @test tuned_model_rbf_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
 
    @test tuned_model_rbf_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
 
    @test tuned_model_rbf_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
 
    n = Threads.nthreads()
    @test tuned_model_rbf_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rbf_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
 
    @test tuned_model_rbf_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rbf_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
 
    @test tuned_model_rbf_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rbf_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)

    # Hyperparameters values verification
    @test tuned_model_rbf_0.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_0.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_0.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_0.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_0.model.model.batch_size == batch_size

    @test tuned_model_rbf_1.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_1.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_1.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_1.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_1.model.model.batch_size == batch_size

    @test tuned_model_rbf_2.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_2.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_2.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_2.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_2.model.model.batch_size == batch_size

    @test tuned_model_rbf_3.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_3.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_3.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_3.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_3.model.model.batch_size == batch_size

    @test tuned_model_rbf_4.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_4.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_4.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_4.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_4.model.model.batch_size == batch_size

    @test tuned_model_rbf_5.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_5.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_5.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_5.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_5.model.model.batch_size == batch_size

    @test tuned_model_rbf_6.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_6.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_6.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_6.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_6.model.model.batch_size == batch_size

    @test tuned_model_rbf_7.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_7.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_7.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_7.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_7.model.model.batch_size == batch_size

    @test tuned_model_rbf_8.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_8.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_8.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_8.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_8.model.model.batch_size == batch_size

    @test tuned_model_rbf_9.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_9.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_9.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_9.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_9.model.model.batch_size == batch_size

    @test tuned_model_rbf_10.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_10.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_10.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_10.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_10.model.model.batch_size == batch_size

    @test tuned_model_rbf_11.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_11.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_11.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_11.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_11.model.model.batch_size == batch_size

    @test tuned_model_rbf_12.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_12.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_12.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_12.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_12.model.model.batch_size == batch_size

    @test tuned_model_rbf_13.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_13.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_13.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_13.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_13.model.model.batch_size == batch_size

    @test tuned_model_rbf_14.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_14.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_14.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_14.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_14.model.model.batch_size == batch_size

    @test tuned_model_rbf_15.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_15.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_15.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_15.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_15.model.model.batch_size == batch_size

    @test tuned_model_rbf_16.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_16.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_16.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_16.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_16.model.model.batch_size == batch_size

    @test tuned_model_rbf_17.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_17.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_17.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_17.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_17.model.model.batch_size == batch_size

    @test tuned_model_rbf_18.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_18.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_18.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_18.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_18.model.model.batch_size == batch_size

    @test tuned_model_rbf_19.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_19.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_19.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_19.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_19.model.model.batch_size == batch_size

    @test tuned_model_rbf_20.model.range[1].lower == minimum_neuron
    @test tuned_model_rbf_20.model.range[1].upper == maximum_neuron
    @test tuned_model_rbf_20.model.range[2].lower == minimum_epochs
    @test tuned_model_rbf_20.model.range[2].upper == maximum_epochs
    @test tuned_model_rbf_20.model.model.batch_size == batch_size

end

@testset "Neural Network Builder rnn modification algorithms" begin

    architecture = "rnn"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_rnn_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_rnn_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_rnn_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_rnn_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rnn_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rnn_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_rnn_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    architecture = "rnn"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_rnn_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_rnn_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_rnn_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_rnn_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rnn_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rnn_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_rnn_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    architecture = "rnn"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_rnn_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_rnn_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_rnn_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_rnn_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_rnn_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_rnn_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_rnn_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_rnn_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rnn_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rnn_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rnn_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rnn_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rnn_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rnn_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_rnn_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rnn_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rnn_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rnn_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rnn_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rnn_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rnn_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_rnn_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_rnn_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_rnn_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_rnn_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_rnn_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_rnn_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_rnn_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_rnn_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_rnn_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_rnn_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_rnn_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_rnn_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_rnn_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_rnn_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_rnn_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_rnn_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_rnn_0.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_0.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_0.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_0.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_0.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_0.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_0.model.model.batch_size == batch_size
    @test tuned_model_rnn_0.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_1.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_1.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_1.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_1.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_1.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_1.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_1.model.model.batch_size == batch_size
    @test tuned_model_rnn_1.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_2.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_2.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_2.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_2.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_2.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_2.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_2.model.model.batch_size == batch_size
    @test tuned_model_rnn_2.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_3.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_3.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_3.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_3.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_3.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_3.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_3.model.model.batch_size == batch_size
    @test tuned_model_rnn_3.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_4.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_4.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_4.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_4.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_4.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_4.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_4.model.model.batch_size == batch_size
    @test tuned_model_rnn_4.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_5.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_5.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_5.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_5.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_5.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_5.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_5.model.model.batch_size == batch_size
    @test tuned_model_rnn_5.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_6.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_6.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_6.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_6.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_6.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_6.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_6.model.model.batch_size == batch_size
    @test tuned_model_rnn_6.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_7.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_7.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_7.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_7.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_7.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_7.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_7.model.model.batch_size == batch_size
    @test tuned_model_rnn_7.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_8.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_8.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_8.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_8.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_8.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_8.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_8.model.model.batch_size == batch_size
    @test tuned_model_rnn_8.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_9.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_9.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_9.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_9.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_9.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_9.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_9.model.model.batch_size == batch_size
    @test tuned_model_rnn_9.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_10.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_10.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_10.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_10.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_10.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_10.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_10.model.model.batch_size == batch_size
    @test tuned_model_rnn_10.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_11.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_11.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_11.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_11.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_11.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_11.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_11.model.model.batch_size == batch_size
    @test tuned_model_rnn_11.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_12.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_12.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_12.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_12.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_12.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_12.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_12.model.model.batch_size == batch_size
    @test tuned_model_rnn_12.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_13.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_13.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_13.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_13.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_13.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_13.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_13.model.model.batch_size == batch_size
    @test tuned_model_rnn_13.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_14.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_14.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_14.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_14.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_14.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_14.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_14.model.model.batch_size == batch_size
    @test tuned_model_rnn_14.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_15.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_15.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_15.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_15.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_15.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_15.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_15.model.model.batch_size == batch_size
    @test tuned_model_rnn_15.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_16.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_16.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_16.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_16.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_16.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_16.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_16.model.model.batch_size == batch_size
    @test tuned_model_rnn_16.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_17.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_17.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_17.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_17.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_17.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_17.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_17.model.model.batch_size == batch_size
    @test tuned_model_rnn_17.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_18.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_18.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_18.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_18.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_18.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_18.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_18.model.model.batch_size == batch_size
    @test tuned_model_rnn_18.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_19.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_19.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_19.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_19.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_19.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_19.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_19.model.model.batch_size == batch_size
    @test tuned_model_rnn_19.model.model.builder.σ == Flux.swish

    @test tuned_model_rnn_20.model.range[1].lower == minimum_neuron
    @test tuned_model_rnn_20.model.range[1].upper == maximum_neuron
    @test tuned_model_rnn_20.model.range[2].lower == minimum_layers
    @test tuned_model_rnn_20.model.range[2].upper == maximum_layers
    @test tuned_model_rnn_20.model.range[3].lower == minimum_epochs
    @test tuned_model_rnn_20.model.range[3].upper == maximum_epochs
    @test tuned_model_rnn_20.model.model.batch_size == batch_size
    @test tuned_model_rnn_20.model.model.builder.σ == Flux.swish

end


@testset "Neural Network Builder lstm modification algorithms" begin

    architecture = "lstm"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_lstm_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_lstm_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_lstm_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_lstm_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_lstm_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_lstm_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_lstm_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    architecture = "lstm"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_lstm_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_lstm_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_lstm_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_lstm_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_lstm_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_lstm_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_lstm_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    architecture = "lstm"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_lstm_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_lstm_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_lstm_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_lstm_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_lstm_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_lstm_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_lstm_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_lstm_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_lstm_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_lstm_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_lstm_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_lstm_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_lstm_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_lstm_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_lstm_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_lstm_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_lstm_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_lstm_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_lstm_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_lstm_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_lstm_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_lstm_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_lstm_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_lstm_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_lstm_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_lstm_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_lstm_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_lstm_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_lstm_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_lstm_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_lstm_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_lstm_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_lstm_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_lstm_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_lstm_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_lstm_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_lstm_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_lstm_0.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_0.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_0.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_0.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_0.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_0.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_0.model.model.batch_size == batch_size
    @test tuned_model_lstm_0.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_1.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_1.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_1.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_1.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_1.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_1.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_1.model.model.batch_size == batch_size
    @test tuned_model_lstm_1.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_2.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_2.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_2.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_2.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_2.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_2.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_2.model.model.batch_size == batch_size
    @test tuned_model_lstm_2.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_3.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_3.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_3.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_3.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_3.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_3.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_3.model.model.batch_size == batch_size
    @test tuned_model_lstm_3.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_4.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_4.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_4.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_4.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_4.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_4.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_4.model.model.batch_size == batch_size
    @test tuned_model_lstm_4.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_5.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_5.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_5.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_5.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_5.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_5.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_5.model.model.batch_size == batch_size
    @test tuned_model_lstm_5.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_6.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_6.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_6.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_6.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_6.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_6.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_6.model.model.batch_size == batch_size
    @test tuned_model_lstm_6.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_7.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_7.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_7.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_7.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_7.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_7.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_7.model.model.batch_size == batch_size
    @test tuned_model_lstm_7.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_8.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_8.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_8.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_8.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_8.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_8.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_8.model.model.batch_size == batch_size
    @test tuned_model_lstm_8.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_9.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_9.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_9.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_9.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_9.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_9.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_9.model.model.batch_size == batch_size
    @test tuned_model_lstm_9.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_10.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_10.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_10.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_10.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_10.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_10.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_10.model.model.batch_size == batch_size
    @test tuned_model_lstm_10.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_11.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_11.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_11.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_11.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_11.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_11.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_11.model.model.batch_size == batch_size
    @test tuned_model_lstm_11.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_12.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_12.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_12.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_12.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_12.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_12.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_12.model.model.batch_size == batch_size
    @test tuned_model_lstm_12.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_13.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_13.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_13.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_13.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_13.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_13.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_13.model.model.batch_size == batch_size
    @test tuned_model_lstm_13.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_14.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_14.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_14.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_14.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_14.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_14.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_14.model.model.batch_size == batch_size
    @test tuned_model_lstm_14.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_15.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_15.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_15.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_15.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_15.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_15.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_15.model.model.batch_size == batch_size
    @test tuned_model_lstm_15.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_16.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_16.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_16.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_16.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_16.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_16.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_16.model.model.batch_size == batch_size
    @test tuned_model_lstm_16.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_17.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_17.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_17.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_17.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_17.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_17.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_17.model.model.batch_size == batch_size
    @test tuned_model_lstm_17.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_18.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_18.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_18.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_18.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_18.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_18.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_18.model.model.batch_size == batch_size
    @test tuned_model_lstm_18.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_19.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_19.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_19.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_19.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_19.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_19.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_19.model.model.batch_size == batch_size
    @test tuned_model_lstm_19.model.model.builder.σ == Flux.swish

    @test tuned_model_lstm_20.model.range[1].lower == minimum_neuron
    @test tuned_model_lstm_20.model.range[1].upper == maximum_neuron
    @test tuned_model_lstm_20.model.range[2].lower == minimum_layers
    @test tuned_model_lstm_20.model.range[2].upper == maximum_layers
    @test tuned_model_lstm_20.model.range[3].lower == minimum_epochs
    @test tuned_model_lstm_20.model.range[3].upper == maximum_epochs
    @test tuned_model_lstm_20.model.model.batch_size == batch_size
    @test tuned_model_lstm_20.model.model.builder.σ == Flux.swish

end

@testset "Neural Network Builder gru modification algorithms" begin

    architecture = "gru"
    processor = "cpu_1"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_gru_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_gru_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_gru_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_gru_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_gru_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_gru_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_gru_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    architecture = "gru"
    processor = "cpu_threads"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_gru_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_gru_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_gru_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_gru_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_gru_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_gru_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_gru_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    architecture = "gru"
    processor = "cpu_processes"
    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_gru_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_gru_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_gru_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_gru_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_gru_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_gru_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_gru_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            fraction_train = 0.8,
        ) #dispatched function

    # Algorithm verification
    @test typeof(tuned_model_gru_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_gru_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_gru_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_gru_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_gru_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_gru_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_gru_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_gru_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_gru_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_gru_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_gru_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_gru_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_gru_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_gru_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_gru_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_gru_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_gru_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_gru_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_gru_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_gru_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_gru_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_gru_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_gru_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_gru_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_gru_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_gru_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_gru_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_gru_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_gru_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_gru_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_gru_0.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_0.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_0.model.range[2].lower == minimum_layers
    @test tuned_model_gru_0.model.range[2].upper == maximum_layers
    @test tuned_model_gru_0.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_0.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_0.model.model.batch_size == batch_size
    @test tuned_model_gru_0.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_1.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_1.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_1.model.range[2].lower == minimum_layers
    @test tuned_model_gru_1.model.range[2].upper == maximum_layers
    @test tuned_model_gru_1.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_1.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_1.model.model.batch_size == batch_size
    @test tuned_model_gru_1.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_2.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_2.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_2.model.range[2].lower == minimum_layers
    @test tuned_model_gru_2.model.range[2].upper == maximum_layers
    @test tuned_model_gru_2.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_2.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_2.model.model.batch_size == batch_size
    @test tuned_model_gru_2.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_3.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_3.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_3.model.range[2].lower == minimum_layers
    @test tuned_model_gru_3.model.range[2].upper == maximum_layers
    @test tuned_model_gru_3.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_3.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_3.model.model.batch_size == batch_size
    @test tuned_model_gru_3.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_4.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_4.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_4.model.range[2].lower == minimum_layers
    @test tuned_model_gru_4.model.range[2].upper == maximum_layers
    @test tuned_model_gru_4.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_4.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_4.model.model.batch_size == batch_size
    @test tuned_model_gru_4.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_5.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_5.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_5.model.range[2].lower == minimum_layers
    @test tuned_model_gru_5.model.range[2].upper == maximum_layers
    @test tuned_model_gru_5.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_5.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_5.model.model.batch_size == batch_size
    @test tuned_model_gru_5.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_6.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_6.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_6.model.range[2].lower == minimum_layers
    @test tuned_model_gru_6.model.range[2].upper == maximum_layers
    @test tuned_model_gru_6.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_6.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_6.model.model.batch_size == batch_size
    @test tuned_model_gru_6.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_7.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_7.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_7.model.range[2].lower == minimum_layers
    @test tuned_model_gru_7.model.range[2].upper == maximum_layers
    @test tuned_model_gru_7.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_7.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_7.model.model.batch_size == batch_size
    @test tuned_model_gru_7.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_8.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_8.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_8.model.range[2].lower == minimum_layers
    @test tuned_model_gru_8.model.range[2].upper == maximum_layers
    @test tuned_model_gru_8.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_8.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_8.model.model.batch_size == batch_size
    @test tuned_model_gru_8.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_9.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_9.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_9.model.range[2].lower == minimum_layers
    @test tuned_model_gru_9.model.range[2].upper == maximum_layers
    @test tuned_model_gru_9.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_9.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_9.model.model.batch_size == batch_size
    @test tuned_model_gru_9.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_10.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_10.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_10.model.range[2].lower == minimum_layers
    @test tuned_model_gru_10.model.range[2].upper == maximum_layers
    @test tuned_model_gru_10.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_10.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_10.model.model.batch_size == batch_size
    @test tuned_model_gru_10.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_11.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_11.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_11.model.range[2].lower == minimum_layers
    @test tuned_model_gru_11.model.range[2].upper == maximum_layers
    @test tuned_model_gru_11.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_11.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_11.model.model.batch_size == batch_size
    @test tuned_model_gru_11.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_12.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_12.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_12.model.range[2].lower == minimum_layers
    @test tuned_model_gru_12.model.range[2].upper == maximum_layers
    @test tuned_model_gru_12.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_12.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_12.model.model.batch_size == batch_size
    @test tuned_model_gru_12.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_13.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_13.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_13.model.range[2].lower == minimum_layers
    @test tuned_model_gru_13.model.range[2].upper == maximum_layers
    @test tuned_model_gru_13.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_13.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_13.model.model.batch_size == batch_size
    @test tuned_model_gru_13.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_14.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_14.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_14.model.range[2].lower == minimum_layers
    @test tuned_model_gru_14.model.range[2].upper == maximum_layers
    @test tuned_model_gru_14.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_14.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_14.model.model.batch_size == batch_size
    @test tuned_model_gru_14.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_15.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_15.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_15.model.range[2].lower == minimum_layers
    @test tuned_model_gru_15.model.range[2].upper == maximum_layers
    @test tuned_model_gru_15.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_15.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_15.model.model.batch_size == batch_size
    @test tuned_model_gru_15.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_16.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_16.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_16.model.range[2].lower == minimum_layers
    @test tuned_model_gru_16.model.range[2].upper == maximum_layers
    @test tuned_model_gru_16.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_16.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_16.model.model.batch_size == batch_size
    @test tuned_model_gru_16.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_17.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_17.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_17.model.range[2].lower == minimum_layers
    @test tuned_model_gru_17.model.range[2].upper == maximum_layers
    @test tuned_model_gru_17.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_17.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_17.model.model.batch_size == batch_size
    @test tuned_model_gru_17.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_18.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_18.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_18.model.range[2].lower == minimum_layers
    @test tuned_model_gru_18.model.range[2].upper == maximum_layers
    @test tuned_model_gru_18.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_18.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_18.model.model.batch_size == batch_size
    @test tuned_model_gru_18.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_19.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_19.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_19.model.range[2].lower == minimum_layers
    @test tuned_model_gru_19.model.range[2].upper == maximum_layers
    @test tuned_model_gru_19.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_19.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_19.model.model.batch_size == batch_size
    @test tuned_model_gru_19.model.model.builder.σ == Flux.swish

    @test tuned_model_gru_20.model.range[1].lower == minimum_neuron
    @test tuned_model_gru_20.model.range[1].upper == maximum_neuron
    @test tuned_model_gru_20.model.range[2].lower == minimum_layers
    @test tuned_model_gru_20.model.range[2].upper == maximum_layers
    @test tuned_model_gru_20.model.range[3].lower == minimum_epochs
    @test tuned_model_gru_20.model.range[3].upper == maximum_epochs
    @test tuned_model_gru_20.model.model.batch_size == batch_size
    @test tuned_model_gru_20.model.model.builder.σ == Flux.swish

end

@testset "Neural Network Builder expn modification algorithms" begin

    architecture = "exploration_models"
    processor = "cpu_1"

    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128
    model_exploration = ["Fnn", "ResNet"]

    tuned_model_expn_0 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_expn_1 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_expn_2 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_expn_3 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_expn_4 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_expn_5 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_expn_6 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    processor = "cpu_threads"

    algorithm = "adam"
    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_expn_7 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_expn_8 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_expn_9 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_expn_10 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_expn_11 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_expn_12 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_expn_13 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    processor = "cpu_processes"

    algorithm = "adam"

    maximum_time = Dates.Minute(15)

    activation_function = "swish"
    minimum_epochs = 1
    maximum_epochs = 5000
    minimum_layers = 5
    maximum_layers = 10
    minimum_neuron = 15
    maximum_neuron = 55
    batch_size = 128

    tuned_model_expn_14 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "radam"

    tuned_model_expn_15 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "nadam"

    tuned_model_expn_16 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function
    
    algorithm = "oadam"

    tuned_model_expn_17 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "lbfgs"

    tuned_model_expn_18 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "oaccel"

    tuned_model_expn_19 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

    algorithm = "pso"

    tuned_model_expn_20 = _neural_network_builder(
            ARCHITECTURE_LIST[Symbol(architecture)],
            PROCESSOR_LIST[Symbol(processor)],
            ALGORITHM_LIST[Symbol(algorithm)],
            maximum_time;
            neuralnet_activation_function = activation_function,
            neuralnet_minimum_epochs = minimum_epochs,
            neuralnet_maximum_epochs = maximum_epochs,
            neuralnet_minimum_layers = minimum_layers,
            neuralnet_maximum_layers = maximum_layers,
            neuralnet_minimum_neuron = minimum_neuron,
            neuralnet_maximum_neuron = maximum_neuron,
            neuralnet_batch_size = batch_size,
            model_exploration = model_exploration,
        ) #dispatched function

        

    # Algorithm verification
    @test typeof(tuned_model_expn_0.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_expn_1.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_expn_2.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_expn_3.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_expn_4.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_expn_5.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_expn_6.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_expn_7.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_expn_8.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_expn_9.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_expn_10.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_expn_11.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_expn_12.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_expn_13.model.model.optimiser) == Optim.ParticleSwarm{Any}

    @test typeof(tuned_model_expn_14.model.model.optimiser) == Flux.Optimise.Adam
    @test typeof(tuned_model_expn_15.model.model.optimiser) == Flux.Optimise.RAdam
    @test typeof(tuned_model_expn_16.model.model.optimiser) == Flux.Optimise.NAdam
    @test typeof(tuned_model_expn_17.model.model.optimiser) == Flux.Optimise.OAdam
    @test typeof(tuned_model_expn_18.model.model.optimiser) == Optim.LBFGS{Nothing, LineSearches.InitialStatic{Float64}, LineSearches.HagerZhang{Float64, Base.RefValue{Bool}}, Optim.var"#19#21"}
    @test typeof(tuned_model_expn_19.model.model.optimiser) == Optim.OACCEL{InitialStatic{Float64}, Float64, GradientDescent{InitialStatic{Float64}, Static, Nothing, Optim.var"#14#16"}, HagerZhang{Float64, Base.RefValue{Bool}}}
    @test typeof(tuned_model_expn_20.model.model.optimiser) == Optim.ParticleSwarm{Any}

    # Processors verification
    @test tuned_model_expn_0.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_1.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_2.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_3.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_4.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_5.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_6.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_expn_7.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_8.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_9.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_10.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_11.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_12.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_13.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_expn_14.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_15.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_16.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_17.model.model.acceleration == CUDALibs{Nothing}(nothing) || ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_18.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_19.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_20.model.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    n = Threads.nthreads()
    @test tuned_model_expn_0.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_1.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_2.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_3.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_4.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_5.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)
    @test tuned_model_expn_6.model.acceleration == ComputationalResources.CPU1{Nothing}(nothing)

    @test tuned_model_expn_7.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_8.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_9.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_10.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_11.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_12.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)
    @test tuned_model_expn_13.model.acceleration == ComputationalResources.CPUThreads{Int64}(n)

    @test tuned_model_expn_14.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_15.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_16.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_17.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_18.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_19.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)
    @test tuned_model_expn_20.model.acceleration == ComputationalResources.CPUProcesses{Nothing}(nothing)


    # Hyperparameters values verification
    @test tuned_model_expn_0.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_0.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_0.model.range[2].lower == minimum_layers
    @test tuned_model_expn_0.model.range[2].upper == maximum_layers
    @test tuned_model_expn_0.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_0.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_0.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_0.model.model.batch_size == batch_size
    @test tuned_model_expn_0.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_1.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_1.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_1.model.range[2].lower == minimum_layers
    @test tuned_model_expn_1.model.range[2].upper == maximum_layers
    @test tuned_model_expn_1.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_1.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_1.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_1.model.model.batch_size == batch_size
    @test tuned_model_expn_1.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_2.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_2.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_2.model.range[2].lower == minimum_layers
    @test tuned_model_expn_2.model.range[2].upper == maximum_layers
    @test tuned_model_expn_2.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_2.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_2.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_2.model.model.batch_size == batch_size
    @test tuned_model_expn_2.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_3.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_3.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_3.model.range[2].lower == minimum_layers
    @test tuned_model_expn_3.model.range[2].upper == maximum_layers
    @test tuned_model_expn_3.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_3.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_3.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_3.model.model.batch_size == batch_size
    @test tuned_model_expn_3.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_4.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_4.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_4.model.range[2].lower == minimum_layers
    @test tuned_model_expn_4.model.range[2].upper == maximum_layers
    @test tuned_model_expn_4.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_4.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_4.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_4.model.model.batch_size == batch_size
    @test tuned_model_expn_4.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_5.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_5.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_5.model.range[2].lower == minimum_layers
    @test tuned_model_expn_5.model.range[2].upper == maximum_layers
    @test tuned_model_expn_5.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_5.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_5.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_5.model.model.batch_size == batch_size
    @test tuned_model_expn_5.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_6.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_6.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_6.model.range[2].lower == minimum_layers
    @test tuned_model_expn_6.model.range[2].upper == maximum_layers
    @test tuned_model_expn_6.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_6.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_6.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_6.model.model.batch_size == batch_size
    @test tuned_model_expn_6.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_7.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_7.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_7.model.range[2].lower == minimum_layers
    @test tuned_model_expn_7.model.range[2].upper == maximum_layers
    @test tuned_model_expn_7.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_7.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_7.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_7.model.model.batch_size == batch_size
    @test tuned_model_expn_7.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_8.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_8.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_8.model.range[2].lower == minimum_layers
    @test tuned_model_expn_8.model.range[2].upper == maximum_layers
    @test tuned_model_expn_8.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_8.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_8.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_8.model.model.batch_size == batch_size
    @test tuned_model_expn_8.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_9.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_9.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_9.model.range[2].lower == minimum_layers
    @test tuned_model_expn_9.model.range[2].upper == maximum_layers
    @test tuned_model_expn_9.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_9.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_9.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_9.model.model.batch_size == batch_size
    @test tuned_model_expn_9.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_10.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_10.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_10.model.range[2].lower == minimum_layers
    @test tuned_model_expn_10.model.range[2].upper == maximum_layers
    @test tuned_model_expn_10.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_10.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_10.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_10.model.model.batch_size == batch_size
    @test tuned_model_expn_10.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_11.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_11.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_11.model.range[2].lower == minimum_layers
    @test tuned_model_expn_11.model.range[2].upper == maximum_layers
    @test tuned_model_expn_11.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_11.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_11.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_11.model.model.batch_size == batch_size
    @test tuned_model_expn_11.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_12.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_12.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_12.model.range[2].lower == minimum_layers
    @test tuned_model_expn_12.model.range[2].upper == maximum_layers
    @test tuned_model_expn_12.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_12.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_12.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_12.model.model.batch_size == batch_size
    @test tuned_model_expn_12.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_13.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_13.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_13.model.range[2].lower == minimum_layers
    @test tuned_model_expn_13.model.range[2].upper == maximum_layers
    @test tuned_model_expn_13.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_13.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_13.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_13.model.model.batch_size == batch_size
    @test tuned_model_expn_13.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_14.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_14.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_14.model.range[2].lower == minimum_layers
    @test tuned_model_expn_14.model.range[2].upper == maximum_layers
    @test tuned_model_expn_14.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_14.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_14.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_14.model.model.batch_size == batch_size
    @test tuned_model_expn_14.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_15.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_15.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_15.model.range[2].lower == minimum_layers
    @test tuned_model_expn_15.model.range[2].upper == maximum_layers
    @test tuned_model_expn_15.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_15.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_15.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_15.model.model.batch_size == batch_size
    @test tuned_model_expn_15.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_16.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_16.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_16.model.range[2].lower == minimum_layers
    @test tuned_model_expn_16.model.range[2].upper == maximum_layers
    @test tuned_model_expn_16.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_16.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_16.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_16.model.model.batch_size == batch_size
    @test tuned_model_expn_16.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_17.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_17.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_17.model.range[2].lower == minimum_layers
    @test tuned_model_expn_17.model.range[2].upper == maximum_layers
    @test tuned_model_expn_17.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_17.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_17.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_17.model.model.batch_size == batch_size
    @test tuned_model_expn_17.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_18.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_18.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_18.model.range[2].lower == minimum_layers
    @test tuned_model_expn_18.model.range[2].upper == maximum_layers
    @test tuned_model_expn_18.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_18.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_18.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_18.model.model.batch_size == batch_size
    @test tuned_model_expn_18.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_19.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_19.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_19.model.range[2].lower == minimum_layers
    @test tuned_model_expn_19.model.range[2].upper == maximum_layers
    @test tuned_model_expn_19.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_19.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_19.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_19.model.model.batch_size == batch_size
    @test tuned_model_expn_19.model.model.builder.σ == Flux.swish

    @test tuned_model_expn_20.model.range[1].lower == minimum_neuron
    @test tuned_model_expn_20.model.range[1].upper == maximum_neuron
    @test tuned_model_expn_20.model.range[2].lower == minimum_layers
    @test tuned_model_expn_20.model.range[2].upper == maximum_layers
    @test tuned_model_expn_20.model.range[3].lower == minimum_epochs
    @test tuned_model_expn_20.model.range[3].upper == maximum_epochs
    @test tuned_model_expn_20.model.range[4].values == ("Fnn", "ResNet",)
    @test tuned_model_expn_20.model.model.batch_size == batch_size
    @test tuned_model_expn_20.model.model.builder.σ == Flux.swish

end

end