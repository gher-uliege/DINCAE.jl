using Flux


_Upsample(sz::Tuple, nvar; method = :nearest) = Upsample(method,scale = 2)


function train_init(model,optim; clip_grad = nothing, learning_rate = nothing)
    @assert optim == :ADAM

    opt_chain = (Flux.Adam(learning_rate),)
    if clip_grad !== nothing
        opt_chain = (ClipGrad(clip_grad),opt_chain...)
    end
    opt = Flux.OptimiserChain(opt_chain...)
    opt_state = Flux.setup(opt, model)

    params = Flux.trainables(model)

    @info "number of parameters $(sum(length,params))"
    return (model,params,opt_state)
end


function train_epoch!((model,params,opt_state),dl,learning_rate; clip_grad = nothing)
    Optimisers.adjust!(opt_state, learning_rate)
    loss_sum = 0
    N = 0
    for samples in dl
        loss, grads = Flux.withgradient(model) do m
            loss_function(m,samples...)
        end

        Flux.update!(opt_state, model, grads[1])

        loss_sum += loss
        N += 1
    end

    return loss_sum/N
end

# TODO, exclude biases
weights(m::Chain) = Flux.trainables(m)

weights(c::Conv) = [c.weight]

@inline function _to_device(::Type{Atype}) where Atype <: AbstractArray
    return Flux.cpu
end


