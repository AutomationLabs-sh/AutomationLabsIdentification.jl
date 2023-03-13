# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
    Constraints
Constraints for the trainable paramters of the physical informed dynamical system identification.
"""
struct Constraints
    lower::Vector
    upper::Vector
end

"""
  PhysicsInformed
MLJFlux like builder that constructs a physics informed dynamical system.
"""
mutable struct PhysicsInformed <: MLJFlux.Builder
    t_p::Vector # trainable parameters
    nbr_state::Integer
    nbr_input::Integer
    prob::Any
    constraints::Constraints
end

"""
    physics_informed
Method that constructs a physics informed dynamical system for identification and tuned uer choice parameters.

The following variables are mendatories:
* `f`: A function that mimic the dynamical system.
* `t_p`: The trainable parameters of the function.
* `nbr_state`: The number of state of the dynamical system.
* `nbr_input`: The number of input of the dynamical system.

The following variable is optional:
* `lower_p`: The lower constraints of the trainable parameters.
* `upper_p`: The upper constraints of the trainable parameters.
"""
function physics_informed(
    f::Function,
    t_p::Vector,
    nbr_state::Integer,
    nbr_input::Integer,
    step_time::Float64;
    lower_p::Vector = [-Inf],
    upper_p::Vector = [Inf],
)

    u0 = zeros(nbr_state + nbr_input, 1)

    tspan = (0, step_time)

    prob = DifferentialEquations.ODEProblem(
        f,
        u0,
        tspan,
        t_p,
        save_everystep = false,
        reltol = 1e-9,
        abstol = 1e-9,
        save_start = false,
    )

    cons = Constraints(lower_p, upper_p)

    return PhysicsInformed(t_p, nbr_state, nbr_input, prob, cons)
end

function (m::PhysicsInformed)(in::AbstractVecOrMat)

    function prob_func(prob, i, repeat)
        #return  DifferentialEquations.remake(prob, u0 = in[:, i])
        @. prob.u0 = in[:, i] #+ prob.u0#[:, i] #remake does not work any more, which cause mutable array and Zygote issues ERROR: "No matching function wrapper was found!"
        return prob
    end

    ensemble_prob = DifferentialEquations.EnsembleProblem(m.prob, prob_func = prob_func)

    sol = DifferentialEquations.solve(
        ensemble_prob,
        p = m.t_p,
        DifferentialEquations.EnsembleThreads(),
        trajectories = size(in, 2),
    ) # override with new parameters and new state init and input

    sol_element = map(p -> p.u, sol)

    #return reshape(reduce(hcat, reduce(hcat, sol_element)')', (m.nbr_state + m.nbr_input), :)[1:m.nbr_state,:]
    #return reshape(reduce(vcat,reduce(vcat, sol_element)), (m.nbr_state + m.nbr_input) ,:)[1:m.nbr_state,:]

    return hcat(reduce(vcat, sol_element)...)[1:m.nbr_state, :]

end


#=
function (m::PhysicsInformed)(in::AbstractVecOrMat)

  function predict_rd(u0) # Our 1-layer "neural network"
    DifferentialEquations.solve(m.prob,p=m.t_p, u0=u0)[1:m.nbr_state,:] # override with new parameters and new state init and input
  end

  return predict_rd(in)

end =#


Flux.trainable(m::PhysicsInformed) = (m.t_p) #only t_p are trainable

function Flux.params(m::PhysicsInformed) #dont know why Flux.params does not work!
    return Flux.params(m.t_p)
end

function Base.show(io::IO, l::PhysicsInformed)
    print(io, "PhysicsInformed(state: ", l.nbr_state, ", input: ", l.nbr_input)
    print(io, ", dynamical system: ", l.prob.f.f)
    print(io, ")")
end

function MLJFlux.build(nn::PhysicsInformed, rng, n_in, n_out)

    #nothing to do here 

    return nn
end

"""
  PhysicsInformedOracle
MLJFlux like builder that constructs a physics informed dynamical system with a neural network oracle
"""
mutable struct PhysicsInformedOracle <: MLJFlux.Builder
    physical::PhysicsInformed
    oracle::MLJFlux.Builder
end

"""
    physics_informed_oracle
Method that constructs a physics informed with Oracle dynamical system for identification.

The following variables are mendatories:
* `f`: A function that mimic the dynamical system.
* `nbr_state`: The number of state of the dynamical system.
* `nbr_input`: The number of input of the dynamical system.
* `step_time`: Step time or sample time of data.
* `neural_networks`: The neural network selection for the Oracle.

"""
function physics_informed_oracle(
    f::Function,
    nbr_state::Integer,
    nbr_input::Integer,
    step_time::Float64,
    neural_networks,
)

    #trainable p, no used here, init to zeros vector
    p = [0.0]

    physical_part = physics_informed(f, p, nbr_state, nbr_input, step_time)

    neural_network_part = neural_networks

    return PhysicsInformedOracle(physical_part, neural_network_part)

end

function Base.show(io::IO, l::PhysicsInformedOracle)
    print(
        io,
        "PhysicsInformed(state: ",
        l.physical.nbr_state,
        ", input: ",
        l.physical.nbr_input,
    )
    print(io, ", dynamical system: ", l.physical.prob.f.f, ") + Oracle(")
    print(io, "layer: ", l.oracle.layer, ", neuron: ", l.oracle.neuron)
    print(io, ", activation function: ", l.oracle.σ)
    print(io, ")")
end

function MLJFlux.build(nn::PhysicsInformedOracle, rng, n_in, n_out)

    oracle_nn = MLJFlux.build(nn.oracle, rng, n_in, n_out)

    #the physical part should not have trainable parameters
    return Flux.Parallel(+, nn.physical, oracle_nn)
end
