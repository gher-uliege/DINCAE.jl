using Flux


_Upsample(sz::Tuple, nvar; method = :nearest) = Upsample(method,scale = 2)


function train_init(model,optim; clip_grad = nothing, learning_rate = nothing)
    @assert optim == :ADAM

    opt = Flux.Adam(learning_rate)
    if clip_grad !== nothing
        opt = Flux.Optimiser(ClipValue(clip_grad),opt)
    end
    params = Flux.params(model)

    return (model,params,opt)
end


function train_epoch!((model,params,optim),dl,learning_rate; clip_grad = nothing)
    loss_sum = 0
    N = 0
    for samples in dl
        loss, back = Flux.pullback(params) do
            #model(samples...)
            loss_function(model,samples...)
        end

        grad = back(1f0)
        Flux.update!(optim, params, grad)

        loss_sum += loss
        N += 1
    end

    return loss_sum/N
end

# TODO, exclude biases
weights(m::Chain) = Flux.params(m)

weights(c::Conv) = [c.weight]



@inline function _to_device(::Type{Atype}) where Atype <: CuArray
    return Flux.gpu
end

@inline function _to_device(::Type{Atype}) where Atype <: AbstractArray
    return Flux.cpu
end


