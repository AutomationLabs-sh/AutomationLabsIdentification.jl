# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
module AutomationLabsIdentification

# Package needed
import CUDA
import DataFrames
import DifferentialEquations
import Flux
import DiffEqFlux
import FluxOptTools
import MLJ
import MLJFlux
import MLJMultivariateStatsInterface
import MLJParticleSwarmOptimization
import MLUtils
import MultivariateStats
import Optim
import ProgressMeter
import StableRNGs
import StatsBase
import Zygote
import Dates
import Distributed
import Statistics

# Export data separation
export data_formatting_identification

# Export main function for identification 
export blackbox_identification
export greybox_identification
export proceed_identification

# Include julia files
include("subfunctions/mlj_physical.jl")
include("subfunctions/mlj_model_interface.jl")
include("subfunctions/data_separation.jl")

# Main functions
include("mainfunctions/blackbox_identification.jl")
include("mainfunctions/greybox_identification.jl")
include("mainfunctions/main_identification.jl")

# sub functions black box models
# densenet
include("models/densenet/mlj_neural_densenet.jl")
include("models/densenet/neural_network_builder_densenet.jl")

# resnet 
include("models/resnet/mlj_neural_resnet.jl")
include("models/resnet/neural_network_builder_resnet.jl")

# fnn 
include("models/fnn/mlj_neural_fnn.jl")
include("models/fnn/neural_network_builder_fnn.jl")

# icnn
include("models/icnn/mlj_neural_icnn.jl")
include("models/icnn/neural_network_builder_icnn.jl")

# linear
include("models/linear/neural_network_builder_linear.jl")

# neural net ode type 1
include("models/neuralnetode_type1/mlj_neural_neuralnetode_type1.jl")
include("models/neuralnetode_type1/neural_network_builder_neuralnetODE_type1.jl")

# neural net ode type 2
include("models/neuralnetode_type2/mlj_neural_neuralnetode_type2.jl")
include("models/neuralnetode_type2/neural_network_builder_neuralnetODE_type2.jl")

# polynet
include("models/polynet/mlj_neural_polynet.jl")
include("models/polynet/neural_network_builder_polynet.jl")

# rbf
include("models/rbf/mlj_neural_rbf.jl")
include("models/rbf/neural_network_builder_rbf.jl")

# rnn 
include("models/rnn/mlj_neural_rnn.jl")
include("models/rnn/neural_network_builder_rnn.jl")

# lstm
include("models/lstm/mlj_neural_lstm.jl")
include("models/lstm/neural_network_builder_lstm.jl")

# gru
include("models/gru/mlj_neural_gru.jl")
include("models/gru/neural_network_builder_gru.jl")

# exploration networks 
include("models/explorationnetworks/mlj_neural_explorationnetworks.jl")
include("models/explorationnetworks/neural_network_builder_exploration_networks.jl")

# sub function physics informed
include("subfunctions/physics_informed_builder_oracle.jl")
include("subfunctions/physics_informed_builder_physics.jl")

end
