# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

# Declare a reset! for recurrent neural networks in order to avoid foreach with has issue with FluxOptFlux and Zygote for Optim.LBFGS
# Should be improve with multiple dispatch the fit function and added model to fit! with dispach from the model struct, Fnn, ResNet, ...

import Flux: Recur
function reset_recur!(m::Recur)
    (m.state = m.cell.state0)
end
function reset_recur!(m)
    #nothing to do where it is not a recurrent neural networks
end
function reset!(m)
    for i = 1:1:length(m[2])
        reset_recur!(m[2][i])
    end
end

import Base: length
function length(m::DiffEqFlux.NeuralODE)
    return 0
end

"""
    MLJFlux.fit!(loss, penalty, chain, optimiser::Optim.FirstOrderOptimizer, epochs, verbosity, X, y)
MLJFlux like dispath fit! that train neural networks with Optim.jl's first order methods.
"""
function MLJFlux.fit!(
    loss,
    penalty,
    chain,
    optimiser::Optim.FirstOrderOptimizer,
    epochs,
    verbosity,
    X,
    y,
)

    # Gradient required methods

    # Initialize and start progress meter:
    meter = ProgressMeter.Progress(
        epochs + 1,
        dt = 0,
        desc = "Optimising neural net:",
        barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
        barlen = 25,
        color = :yellow,
    )
    verbosity != 1 || MLJFlux.next!(meter)

    function advanced_time_control(x)
        MLJFlux.next!(meter)
        false
    end

    # Initiate history:
    n_batches = length(y)

    # Refresh zygote
    Zygote.refresh()

    # Get tranable parameters
    parameters = Flux.params(chain)

    # Loss function with Flux reset in case of recurrent neural networks
    #=function losses_recurrent(x, y)
        Flux.reset!(chain)
        l = loss(chain(x), y)
        #Flux.reset!(chain)
        return l
    end

    losses() = Statistics.mean(losses_recurrent(X[i], y[i]) +
                  penalty(parameters)/n_batches for i in 1:n_batches) =#

    #=             function losses_recurrent(x, y)
               #    Flux.reset!(chain)
                   l = loss(chain(x), y)
                   #Flux.reset!(chain)
                   return l
               end

               losses = (losses_recurrent(X[i], y[i]) +
                         penalty(parameters)/n_batches for i in 1:n_batches)=#

    function losses_(X, y)
        #for i in 1 : 1 : length(chain[2])
        #    chain[2][i].state = chain[2][i].cell.state0
        #end
        reset!(chain)
        l = Statistics.mean(
            loss(chain(X[i]), y[i]) + penalty(parameters) for i = 1:n_batches
        )
        return l
    end

    losses() = losses_(X, y) #closure

    # Declare the function loss fonctionnne:
    #losses() =
    #    Statistics.mean(loss(chain(X[i]), y[i]) + penalty(parameters) for i = 1:n_batches)

    # Get gradiant
    lossfun, gradfun, fg!, p0 = FluxOptTools.optfuns(losses, parameters)

    if verbosity == 0
        res = Optim.optimize(
            Optim.only_fg!(fg!),
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
            ),
        )
    elseif verbosity == 1
        res = Optim.optimize(
            Optim.only_fg!(fg!),
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
                callback = advanced_time_control,
            ),
        )
        println(res)
    elseif verbosity >= 2
        res = Optim.optimize(
            Optim.only_fg!(fg!),
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
                show_trace = true,
                show_every = 1,
            ),
        )
        println(res)
    else
        res = Optim.optimize(
            Optim.only_fg!(fg!),
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
            ),
        )
    end

    history = []
    for i = 1:1:size(res.trace, 1)
        push!(history, res.trace[i].value)
    end

    # Add a reset chain in order to allow the evaluate fct to compute the loss from MLJ.TunedModel
    Flux.reset!(chain)

    return chain, history

end

"""
    MLJFlux.fit!(loss, penalty, chain, optimiser::Optim.FirstOrderOptimizer, epochs, verbosity, X, y)
MLJFlux like dispath fit! that train neural networks with Optim.jl's first order methods.
"""

function MLJFlux.fit!(
    loss,
    penalty,
    chain,
    optimiser::Optim.ZerothOrderOptimizer,
    epochs,
    verbosity,
    X,
    y,
)

    # Initialize and start progress meter:
    meter = ProgressMeter.Progress(
        epochs + 1,
        dt = 0,
        desc = "Optimising neural net:",
        barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
        barlen = 25,
        color = :yellow,
    )
    verbosity != 1 || MLJFlux.next!(meter)

    function advanced_time_control(x)
        MLJFlux.next!(meter)
        false
    end

    # Initiate history:
    n_batches = length(y)

    # Get tranable parameters
    parameters = Flux.params(chain)
    p0 = zeros(parameters)
    copy!(p0, parameters)

    # Loss function with Flux reset in case of recurrent neural networks
    function losses_recurrent(x, y)
        Flux.reset!(chain)
        l = loss(chain(x), y)
        #Flux.reset!(chain)
        return l
    end

    function losses(x)

        j = 1
        for p in Flux.params(chain)

            for i = 1:1:length(p)
                p[i] = x[j+i]
            end
            j = j + 1
        end

        return Statistics.mean(
            losses_recurrent(X[i], y[i]) + penalty(parameters) for i = 1:n_batches
        )
    end

    # Declare the function loss
    #=  function losses(x)

          j = 1
          for p in Flux.params(chain)

              for i = 1:1:length(p)
                  p[i] = x[j+i]
              end
              j = j + 1
          end

          return Statistics.mean(
              loss(chain(X[i]), y[i]) + penalty(parameters) for i = 1:n_batches
          )
      end=#

    if verbosity == 0
        res = Optim.optimize(
            losses,
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
            ),
        )
    elseif verbosity == 1
        res = Optim.optimize(
            losses,
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
                callback = advanced_time_control,
            ),
        )
        println(res)
    elseif verbosity >= 2
        res = Optim.optimize(
            losses,
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
                show_trace = true,
                show_every = 1,
            ),
        )
        println(res)
    else
        res = Optim.optimize(
            losses,
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
            ),
        )
    end

    history = []
    for i = 1:1:size(res.trace, 1)
        push!(history, res.trace[i].value)
    end

    # Add a reset chain in order to allow the evaluate fct to compute the loss from MLJ.TunedModel
    Flux.reset!(chain)

    return chain, history

end

"""
    MLJFlux.fit!(loss, penalty, chain::PhysicsInformed, optimiser::Optim.FirstOrderOptimizer, epochs, verbosity, X, y)
MLJFlux like dispath fit! that train physics informed with Optim.jl's first order methods, with or without constraints.
"""
function MLJFlux.fit!(
    loss,
    penalty,
    chain::PhysicsInformed,
    optimiser::Optim.FirstOrderOptimizer,
    epochs,
    verbosity,
    X,
    y,
)

    iteration = 50
    #epoch is incremented with iterated model and Step rather than just adde a iteration and PhysicsInformed
    # To do investigate why.

    # Gradient required methods

    # Initialize and start progress meter:
    meter = ProgressMeter.Progress(
        epochs + 1,
        dt = 0,
        desc = "Optimising neural net:",
        barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
        barlen = 25,
        color = :yellow,
    )
    verbosity != 1 || MLJFlux.next!(meter)

    function advanced_time_control(x)
        MLJFlux.next!(meter)
        false
    end

    # Initiate history:
    n_batches = length(y)

    # Refresh zygote
    Zygote.refresh()

    # Get trainable parameters
    parameters = Flux.params(chain.t_p)

    # Declare the function loss
    losses() =
        Statistics.mean(loss(chain(X[i]), y[i]) + penalty(parameters) for i = 1:n_batches)

    # Get gradiant
    lossfun, gradfun, fg!, p0 = FluxOptTools.optfuns(losses, parameters)

    if chain.constraints.lower == [-Inf] && chain.constraints.upper == [Inf]
        # No constraints with optimisation here  
        if verbosity == 0
            res = Optim.optimize(
                Optim.only_fg!(fg!),
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                ),
            )
        elseif verbosity == 1
            res = Optim.optimize(
                Optim.only_fg!(fg!),
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                    callback = advanced_time_control,
                ),
            )
            println(res)
        elseif verbosity >= 2
            res = Optim.optimize(
                Optim.only_fg!(fg!),
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                    show_trace = true,
                    show_every = 1,
                ),
            )
            println(res)
        else
            res = Optim.optimize(
                Optim.only_fg!(fg!),
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                ),
            )
        end

    else
        # Constraints with optimization here
        inner_optimizer = optimiser
        if verbosity == 0
            res = Optim.optimize(
                Optim.only_fg!(fg!),
                chain.constraints.lower,
                chain.constraints.upper,
                p0,
                Optim.Fminbox(inner_optimizer),
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                ),
            )
        elseif verbosity == 1
            res = Optim.optimize(
                Optim.only_fg!(fg!),
                chain.constraints.lower,
                chain.constraints.upper,
                p0,
                Optim.Fminbox(inner_optimizer),
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                    callback = advanced_time_control,
                ),
            )
            println(res)
        elseif verbosity >= 2
            res = Optim.optimize(
                Optim.only_fg!(fg!),
                chain.constraints.lower,
                chain.constraints.upper,
                p0,
                Optim.Fminbox(inner_optimizer),
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                    show_trace = true,
                    show_every = 1,
                ),
            )
            println(res)
        else
            res = Optim.optimize(
                Optim.only_fg!(fg!),
                chain.constraints.lower,
                chain.constraints.upper,
                p0,
                Optim.Fminbox(inner_optimizer),
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                ),
            )
        end
    end

    history = []
    for i = 1:1:size(res.trace, 1)
        push!(history, res.trace[i].value)
    end

    return chain, history

end

"""
    MLJFlux.fit!(loss, penalty, chain::PhysicsInformed, optimiser::Optim.FirstOrderOptimizer, epochs, verbosity, X, y)
MLJFlux like dispath fit! that train physics informed with Optim.jl's first order methods, with or without constraints.
"""
function MLJFlux.fit!(
    loss,
    penalty,
    chain::PhysicsInformed,
    optimiser::Optim.ZerothOrderOptimizer,
    epochs,
    verbosity,
    X,
    y,
)

    iteration = 50
    #epoch is incremented with iterated model and Step rather than iterated with Step

    # Initialize and start progress meter:
    meter = ProgressMeter.Progress(
        epochs + 1,
        dt = 0,
        desc = "Optimising neural net:",
        barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
        barlen = 25,
        color = :yellow,
    )
    verbosity != 1 || MLJFlux.next!(meter)

    function advanced_time_control(x)
        MLJFlux.next!(meter)
        false
    end

    # Initiate history:
    n_batches = length(y)

    # Get trainable parameters
    parameters = Flux.params(chain.t_p)
    p0 = zeros(parameters)
    copy!(p0, parameters)

    # Declare the function loss
    function losses(x)
        for p in Flux.params(chain.t_p)
            p[:, :] .= x[:, :]
        end
        return Statistics.mean(
            loss(chain(X[i]), y[i]) + penalty(parameters) for i = 1:n_batches
        )
    end

    if chain.constraints.lower == [-Inf] && chain.constraints.upper == [Inf]
        # No constraints with optimisation here  
        if verbosity == 0
            res = Optim.optimize(
                losses,
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                ),
            )
        elseif verbosity == 1
            res = Optim.optimize(
                losses,
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                    callback = advanced_time_control,
                ),
            )
            println(res)
        elseif verbosity >= 2
            res = Optim.optimize(
                losses,
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                    show_trace = true,
                    show_every = 1,
                ),
            )
            println(res)
        else
            res = Optim.optimize(
                losses,
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                ),
            )
        end

    else
        # Constraints with optimization here
        optimiser = Optim.ParticleSwarm(
            lower = chain.constraints.lower,
            upper = chain.constraints.upper,
            n_particles = 0,
        )

        if verbosity == 0
            res = Optim.optimize(
                losses,
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                ),
            )
        elseif verbosity == 1
            res = Optim.optimize(
                losses,
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                    callback = advanced_time_control,
                ),
            )
            println(res)
        elseif verbosity >= 2
            res = Optim.optimize(
                losses,
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                    show_trace = true,
                    show_every = 1,
                ),
            )
            println(res)
        else
            res = Optim.optimize(
                losses,
                p0,
                optimiser,
                Optim.Options(
                    iterations = epochs,
                    allow_f_increases = false,
                    store_trace = true,
                ),
            )
        end
    end

    history = []
    for i = 1:1:size(res.trace, 1)
        push!(history, res.trace[i].value)
    end

    return chain, history

end

#=
"""
    MLJFlux.fit!(loss, penalty, chain, optimiser::Optim.FirstOrderOptimizer, epochs, verbosity, X, y)
MLJFlux like dispath fit! that train neural networks with Optim.jl's first order methods.
"""

function MLJFlux.fit!(
    loss,
    penalty,
    chain,
    optimiser::Optim.ZerothOrderOptimizer,
    epochs,
    verbosity,
    X,
    y,
)

    # Initialize and start progress meter:
    meter = ProgressMeter.Progress(
        epochs + 1,
        dt = 0,
        desc = "Optimising neural net:",
        barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
        barlen = 25,
        color = :yellow,
    )
    verbosity != 1 || MLJFlux.next!(meter)

    function advanced_time_control(x)
        MLJFlux.next!(meter)
        false
    end

    # Initiate history:
    n_batches = length(y)

    # Get tranable parameters
    parameters = Flux.params(chain)
    p0 = zeros(parameters)
    copy!(p0, parameters)

    # Declare the function loss
    function losses_recurrent(x, y)
        Flux.reset!(chain)
        loss(chain(x), y)
    end

    function losses(x)

        j = 1
        for p in Flux.params(chain)

            for i = 1:1:length(p)
                p[i] = x[j+i]
            end
            j = j + 1
        end

        return Statistics.mean(
            losses_recurrent(X[i], y[i]) + penalty(parameters) for i = 1:n_batches
        )
    end

    if verbosity == 0
        res = Optim.optimize(
            losses,
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
            ),
        )
    elseif verbosity == 1
        res = Optim.optimize(
            losses,
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
                callback = advanced_time_control,
            ),
        )
        println(res)
    elseif verbosity >= 2
        res = Optim.optimize(
            losses,
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
                show_trace = true,
                show_every = 1,
            ),
        )
        println(res)
    else
        res = Optim.optimize(
            losses,
            p0,
            optimiser,
            Optim.Options(
                iterations = epochs,
                allow_f_increases = true,
                store_trace = true,
            ),
        )
    end

    history = []
    for i = 1:1:size(res.trace, 1)
        push!(history, res.trace[i].value)
    end

    # Add a reset chain in order to allow the evaluate fct to compute the loss from MLJ.TunedModel
    Flux.reset!(chain)

    return chain, history

end=#


function MLJFlux.fit!(loss, penalty, chain, optimiser, epochs, verbosity, X, y)

    # intitialize and start progress meter:
    meter = ProgressMeter.Progress(
        epochs + 1,
        dt = 0,
        desc = "Optimising neural net:",
        barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
        barlen = 25,
        color = :yellow,
    )
    verbosity != 1 || MLJFlux.next!(meter)

    # initiate history:
    n_batches = length(y)

    parameters = Flux.params(chain)

    # Loss function with Flux reset in case of recurrent neural networks
    function losses_recurrent(x, y)
        Flux.reset!(chain)
        l = loss(chain(x), y)
        #Flux.reset!(chain)
        return l
    end

    losses =
        (losses_recurrent(X[i], y[i]) + penalty(parameters) / n_batches for i = 1:n_batches)
    history = [Statistics.mean(losses)]

    for i = 1:epochs
        current_loss = MLJFlux.train!(loss, penalty, chain, optimiser, X, y)
        verbosity < 2 || @info "Loss is $(round(current_loss; sigdigits=4))"
        verbosity != 1 || MLJFlux.next!(meter)
        push!(history, current_loss)
    end

    # Add a reset chain in order to allow the evaluate fct to compute the loss from MLJ.TunedModel
    Flux.reset!(chain)

    return chain, history
end
