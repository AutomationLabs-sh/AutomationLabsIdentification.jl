# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

### Minimal tests for CI without GPUs ###
print("Testing Architectures...")
took_seconds = @elapsed include("./architectures_neural_test.jl");
println("done (took ", took_seconds, " seconds)")

print("Testing modification algorithm...")
took_seconds = @elapsed include("./neural_network_builder_test.jl");
println("done (took ", took_seconds, " seconds)")

#=
### Extra test for CI without GPUs ###
print("Testing Hyperparameters Optimization...")
took_seconds = @elapsed include("./hyperparameters_optimization_test.jl");
println("done (took ", took_seconds, " seconds)")

print("Testing losses functions...")
took_seconds = @elapsed include("./losses_functions_test.jl");
println("done (took ", took_seconds, " seconds)")

print("Testing QTP identification with linear model and lls algorithm...")
took_seconds = @elapsed include("./qtp_non_iterable_models.jl");
println("done (took ", took_seconds, " seconds)")

print("Testing QTP identification with LBFGS algorithm...")
took_seconds = @elapsed include("./qtp_lbfgs_identification_test.jl");
println("done (took ", took_seconds, " seconds)")

print("Testing QTP identification with OACCEL algorithm...")
took_seconds = @elapsed include("./qtp_oaccel_identification_test.jl");
println("done (took ", took_seconds, " seconds)")

print("Testing QTP identification with Particle Swarm algorithm...")
took_seconds = @elapsed include("./qtp_pso_identification_test.jl");
println("done (took ", took_seconds, " seconds)")


### Test with GPU not suitable github CI ###
print("Testing QTP identification with Back-Propagation algorithm...")
took_seconds = @elapsed include("./qtp_backprop_identification_test.jl");
println("done (took ", took_seconds, " seconds)")

print("Testing blackbox identification...")
took_seconds = @elapsed include("./blackbox_identification_test.jl");
println("done (took ", took_seconds, " seconds)")
=#

### Physics informed learning not robust and need improvement ###

#print("Testing modification algorithm...")
#took_seconds = @elapsed include("./physics_informed_builder_test.jl");
#println("done (took ", took_seconds, " seconds)")

# QTP identification with physics informed
#print("Testing QTP identification with Physics Informed...")
#took_seconds = @elapsed include("./qtp_physics_informed_oracle_test.jl");
#println("done (took ", took_seconds, " seconds)")

#print("Testing greybox identification...")
#took_seconds = @elapsed include("./greybox_identification_test.jl");
#println("done (took ", took_seconds, " seconds)")
