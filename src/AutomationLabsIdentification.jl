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

export extract_model_from_machine
export get_mlj_model_type

# Include julia files

# Types 
include("sub/types.jl")

include("sub/models/physical/mlj_physical.jl")
include("sub/mlj_model_interface.jl")
include("sub/data_separation.jl")

# Main functions
include("main/blackbox_identification.jl")
include("main/greybox_identification.jl")
include("main/main_identification.jl")

# sub functions black box models
# densenet
include("sub/models/densenet/mlj_neural_densenet.jl")
include("sub/models/densenet/neural_network_builder_densenet.jl")

# resnet 
include("sub/models/resnet/mlj_neural_resnet.jl")
include("sub/models/resnet/neural_network_builder_resnet.jl")

# fnn 
include("sub/models/fnn/mlj_neural_fnn.jl")
include("sub/models/fnn/neural_network_builder_fnn.jl")

# icnn
include("sub/models/icnn/mlj_neural_icnn.jl")
include("sub/models/icnn/neural_network_builder_icnn.jl")

# linear
include("sub/models/linear/neural_network_builder_linear.jl")

# neural ode 
include("sub/models/neuralode/mlj_neural_neuralode.jl")
include("sub/models/neuralode/neural_network_builder_neuralode.jl")

# polynet
include("sub/models/polynet/mlj_neural_polynet.jl")
include("sub/models/polynet/neural_network_builder_polynet.jl")

# rbf
include("sub/models/rbf/mlj_neural_rbf.jl")
include("sub/models/rbf/neural_network_builder_rbf.jl")

# rnn 
include("sub/models/rnn/mlj_neural_rnn.jl")
include("sub/models/rnn/neural_network_builder_rnn.jl")

# lstm
include("sub/models/lstm/mlj_neural_lstm.jl")
include("sub/models/lstm/neural_network_builder_lstm.jl")

# gru
include("sub/models/gru/mlj_neural_gru.jl")
include("sub/models/gru/neural_network_builder_gru.jl")

# Rknn1
include("sub/models/rknn1/mlj_neural_rknn1.jl")
include("sub/models/rknn1/neural_network_builder_rknn1.jl")

# Rknn2
include("sub/models/rknn2/mlj_neural_rknn2.jl")
include("sub/models/rknn2/neural_network_builder_rknn2.jl")

# Rknn4
include("sub/models/rknn4/mlj_neural_rknn4.jl")
include("sub/models/rknn4/neural_network_builder_rknn4.jl")

# exploration networks 
include("sub/models/explorationnetworks/mlj_neural_explorationnetworks.jl")
include("sub/models/explorationnetworks/neural_network_builder_exploration_networks.jl")


# sub function physics informed
include("sub/models/physical/physics_informed_builder_oracle.jl")
include("sub/models/physical/physics_informed_builder_physics.jl")

include("sub/extract_models.jl")

end
