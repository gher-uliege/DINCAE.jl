
SumSkip(x) = SkipConnection(x,+)
CatSkip(x) = SkipConnection(x,(mx,y) -> cat(mx, y, dims=Val(3)))


# save inversion
function sinv(x, ; minx = eltype(x)(1e-3))
    T = eltype(x)
    return one(T) ./ max.(x,minx)
end


"""
transform x[:,:,1,:] and x[:,:,2,:] to mean and error variance
"""
function transform_mσ2_single(x,gamma)
    N = ndims(x)
    allst = ntuple(i -> :, N-2)

    loginvσ2_rec = x[allst...,2:2,:]
    invσ2_rec = exp.(min.(loginvσ2_rec,gamma))
    σ2_rec = sinv(invσ2_rec)
    m_rec = x[allst...,1:1,:] .* σ2_rec

    return cat(
       m_rec,
       σ2_rec,
    dims = Val(N-1))
end

function transform_mσ2(x,gamma)
    N = ndims(x)
    allst = ntuple(i -> :, N-2)
    noutput = size(x,N-1) ÷ 2

    pcat(x,y) = cat(x,y,dims = Val(N-1))

    return reduce(
        pcat,
        (transform_mσ2_single(x[allst...,(1:2).+2*ivar,:],gamma) for ivar = 0:noutput-1))
end


function sum_laplacian_penalty(penalty,x)
    if penalty == 0
        return 0
    else
        return penalty * (
            sum((x[3:end,:,:,:] - 2*x[2:end-1,:,:,:] + x[1:end-2,:,:,:]).^2) +
            sum((x[:,3:end,:,:] - 2*x[:,2:end-1,:,:] + x[:,1:end-2,:,:]).^2))
    end
end

function costfun_single(
    m_rec,σ2_rec,m_true,σ2_true,mask_noncloud,truth_uncertain;
    laplacian_penalty = 0,
    laplacian_error_penalty = laplacian_penalty)

    #return sum(m_rec)
    #n_noncloud = sum(mask_noncloud)
    n_noncloud = ChainRulesCore.ignore_derivatives(sum(mask_noncloud))
    #n_noncloud = 100
    # if n_noncloud == 0
    #     @show n_noncloud
    #     error("no data")
    # end
    #@show n_noncloud
    if truth_uncertain
        # KL divergence between two univariate Gaussians p and q
        # p ~ N(σ2_1,\mu_1)
        # q ~ N(σ2_2,\mu_2)
        #
        # 2 KL(p,q) = log(σ2_2/σ2_1) + (σ2_1 + (μ_1 - μ_2)^2)/(σ2_2) - 1
        # 2 KL(p,q) = log(σ2_2) - log(σ2_1) + (σ2_1 + (μ_1 - μ_2)^2)/(σ2_2) - 1
        # 2 KL(p_true,q_rec) = log(σ2_rec/σ2_true) + (σ2_true + (μ_rec - μ_true)^2)/(σ2_rec) - 1

        # where there is cloud, σ2_ratio is 1 and its log is zero

        σ2_ratio = (σ2_rec ./ σ2_true) .* mask_noncloud + (1 .- mask_noncloud)
        #cost = (sum(log.(σ2_ratio)) + sum((σ2_true + difference.^2) ./ σ2_rec)) / n_noncloud

        σ2_rec_noncloud = σ2_rec .* mask_noncloud + (1 .- mask_noncloud)
        difference2 = ((m_rec - m_true).^2  + σ2_true)  .* mask_noncloud

        #cost = (sum(log.(σ2_rec_noncloud)) + sum(difference2 ./ σ2_rec)) / n_noncloud
        cost = (sum(log.(σ2_ratio)) + sum(difference2 ./ σ2_rec)) / n_noncloud
    else
        # where there is cloud, σ2_rec_noncloud is 1 and its log is zero
        σ2_rec_noncloud = σ2_rec .* mask_noncloud + (1 .- mask_noncloud)
        difference2 = (m_rec - m_true).^2  .* mask_noncloud

        cost = (sum(log.(σ2_rec_noncloud)) + sum(difference2 ./ σ2_rec)) / n_noncloud
    end

    #return sum((m_rec - m_true).^2  .* mask_noncloud)
    #TEST

    return (cost
            + sum_laplacian_penalty(laplacian_penalty,m_rec)
            + sum_laplacian_penalty(laplacian_error_penalty,σ2_rec))
end
function costfun_single(xrec,xtrue,truth_uncertain; kwargs...)
    N = ndims(xtrue)
    allst = ntuple(i -> :, N-2)

    m_rec = xrec[allst...,1:1,:]
    σ2_rec = xrec[allst...,2:2,:]

    σ2_true = sinv(xtrue[allst...,2:2,:])
    m_true = xtrue[allst...,1:1,:] .* σ2_true

    # # 1 if measurement
    # # 0 if no measurement (cloud or land for SST)
    mask_noncloud = xtrue[allst...,2:2,:] .!= 0

#    @show extrema(Array(m_rec))
#    @show extrema(Array(m_true))

    cost = costfun_single(m_rec,σ2_rec,m_true,σ2_true,mask_noncloud,
                          truth_uncertain; kwargs...)

    #difference2 = (m_rec - m_true).^2  .* mask_noncloud
    #costMSE = sum(difference2) / n_noncloud

    #return costMSE
    #    cost = sum(σ2_rec .* mask_noncloud)

    return cost
end

# cost function for multivariate output case
function costfun(xrec,xtrue,truth_uncertain; kwargs...)
    N = ndims(xtrue)
    noutput = size(xtrue,N-1) ÷ 2

    allst = ntuple(i -> :, N-2)

    # sum over all output parameters
    return sum(
        [
            costfun_single(
                xrec[allst...,(1:2) .+ 2*ivar,:],
                xtrue[allst...,(1:2) .+ 2*ivar,:],
                truth_uncertain;
                kwargs...
            )
            for ivar in 0:noutput-1 ])
end

# Model
struct StepModel{F1,F2}
    chains
    loss_weights
    regularization_L1
    regularization_L2
    final_layer::F1
    costfun::F2
end

weights(model::StepModel) = Iterators.flatten(weights.(model.chains))

function StepModel(
    chains,loss_weights,truth_uncertain,gamma;
    regularization_L1 = 0.f0,
    regularization_L2 = 0.f0,
    laplacian_penalty = 0,
    laplacian_error_penalty = laplacian_penalty,
    final_layer = xout -> transform_mσ2(xout,gamma),
    costfun = (xout,xtrue) -> costfun(
        xout,xtrue,truth_uncertain,
        laplacian_penalty = laplacian_penalty,
        laplacian_error_penalty = laplacian_error_penalty),
    )
    return StepModel(
        chains,loss_weights,
        regularization_L1, regularization_L2,
        final_layer,
        costfun,
    )
end

function (model::StepModel)(xin)
    xout = model.chains[1](xin)

    for i = 2:length(model.chains)
        xout = model.chains[i](cat(xout,xin,dims=3))
    end
    return model.final_layer(xout)
end

#function (model::StepModel)(xin,xtrue)
function loss_function(model::StepModel,xin,xtrue)
#    loss = sum(model(xin))
    N = ndims(xin)
    noutput = size(xtrue,N-1) ÷ 2
    xout = model.chains[1](xin)

    loss = model.loss_weights[1] *
        model.costfun(model.final_layer(xout),xtrue)

    #@show loss
    for i = 2:length(model.chains)
        xout = model.chains[i](cat(xout,xin,dims=3))
        loss += model.loss_weights[i] *
            model.costfun(model.final_layer(xout),xtrue)
        #@show loss
    end

    # regularization
    for w in weights(model)
        if model.regularization_L1 !== 0
            #@show i,j,model.regularization_L1
            loss += model.regularization_L1 * sum(abs,w)
        end
        if model.regularization_L2 !== 0
            #@show model.regularization_L2,size(w)
            loss += model.regularization_L2 * sum(abs2,w)
        end
    end

    return loss
end

Flux.@layer StepModel

function showsize(x)
    @show size(x)
    return x
end

# remove padding from x
function croppadding(x,odd)
    r = ntuple(i -> 1:(size(x,i)-odd[i]), length(odd))
    return x[r...,:,:]
end

function croppadding2(x,odd)
    r = ntuple(i -> ((odd[i]==1) ? (1:size(x,i)-1) : (2:size(x,i)-1) ), length(odd))
    return x[r...,:,:]
end

function recmodel4(sz,enc_nfilter,dec_nfilter,skipconnections,l=1; method = :nearest)
    # for 2D
    # convkernel = (3,3)
    # upkernel = (2,2)

    convkernel = ntuple(i -> 3, length(sz))
    upkernel = ntuple(i -> 2, length(sz))

    # activation function
    f =
        if l == 1
            identity
        else
            relu
        end

    odd = map(x -> x % 2,sz)

    # size after pooling
    sz_small = sz.÷2 .+ odd

    if l == length(enc_nfilter)
        return identity
    else
        inner = Chain(
            Conv(convkernel,enc_nfilter[l] => enc_nfilter[l+1],relu, pad = 1),
            MeanPool(upkernel,pad = odd),
            recmodel4(sz_small,enc_nfilter,dec_nfilter,skipconnections,l+1, method = method),
            _Upsample(sz_small, enc_nfilter[l+1], method = method),
            x -> croppadding(x,odd),
            Conv(convkernel,dec_nfilter[l+1] => dec_nfilter[l],f, pad = 1),
#            ConvTranspose(upkernel,dec_nfilter[l+1] => dec_nfilter[l],f,stride=2),
#            x -> croppadding(x,odd),
        )

        if l in skipconnections
            println("skip connections at level $l")
            return SumSkip(inner)
        else
            return inner
        end
    end
end

function genmodel(sz,ninput,noutput;
                  truth_uncertain = false,
                  enc_nfilter_internal = [16,24,36,54],
                  skipconnections = 2:(length(enc_nfilter_internal)+1),
                  regularization_L1_beta = 0,
                  regularization_L2_beta = 0,
                  upsampling_method = :nearest,
                  loss_weights_refine = (1.,),
                  min_std_err = 0.006737946999085467,
                  output_ndims = 1,
                  direction_obs = nothing,
                  laplacian_penalty = 0,
                  laplacian_error_penalty = laplacian_penalty,
                  )

    nvar = ninput
    enc_nfilter = vcat([nvar],enc_nfilter_internal)

    if output_ndims == 1
        dec_nfilter = vcat([2*noutput],enc_nfilter_internal)
    else
        noutput = output_ndims
        # 5 (u,v) and lower matrix of P
        dec_nfilter = vcat([nscalar_per_obs_(output_ndims)],
                           enc_nfilter_internal)
    end


    @info "Number of filters in encoder: $enc_nfilter"
    @info "Number of filters in decoder: $dec_nfilter"


    gamma = log(min_std_err^(-2))
    @info "Gamma:             $gamma"

    @info "Number of filters: $enc_nfilter"
    if loss_weights_refine == (1.,)
        steps = (DINCAE.recmodel4(
            sz,
                enc_nfilter,
                dec_nfilter,
                skipconnections,
                method = upsampling_method),)
    else
        println("Step model")

        enc_nfilter2 = copy(enc_nfilter)
        enc_nfilter2[1] += dec_nfilter[1]
        dec_nfilter2 = copy(dec_nfilter)
        @info "Number of filters in encoder (refinement): $enc_nfilter2"
        @info "Number of filters in decoder (refinement): $dec_nfilter2"

        steps = (DINCAE.recmodel4(sz,enc_nfilter,dec_nfilter,skipconnections; method = upsampling_method),
                 DINCAE.recmodel4(sz,enc_nfilter2,dec_nfilter2,skipconnections; method = upsampling_method))
    end

    if output_ndims == 1
        model = StepModel(
            steps,loss_weights_refine,truth_uncertain,gamma;
            regularization_L1 = regularization_L1_beta,
            regularization_L2 = regularization_L2_beta,
            laplacian_penalty,
            laplacian_error_penalty,
        )
    else
        model = StepModel(
            steps,loss_weights_refine,truth_uncertain,gamma;
            final_layer = identity,
            costfun = (xrec,xtrue) -> vector2_costfun(xrec,xtrue,truth_uncertain,direction_obs),
            regularization_L1 = regularization_L1_beta,
            regularization_L2 = regularization_L2_beta,
            laplacian_penalty,
            laplacian_error_penalty,
        )
    end

    return model
end


function loadmodel(model_fname::AbstractString; device=cpu)
    function apply_model(xin)
        xrec = Array(model(device(xin)))

        return (train_mean_data .+ xrec[:,:,1,:],
                sqrt.(xrec[:,:,2,:]))
    end

    JLD2.@load(
        model_fname,model_state,ntime_win,is3D,train_mean_data,
        cycle_periods,remove_mean,direction_obs,e,noutput,enc_nfilter_internal,
        upsampling_method,skipconnections,min_std_err,loss_weights_refine,
        truth_uncertain,output_ndims,
        regularization_L1_beta,regularization_L2_beta,
        laplacian_penalty,laplacian_error_penalty,input_size)

    sz = input_size
    model = genmodel(sz[1:end-1],sz[end],noutput;
                     enc_nfilter_internal,
                     upsampling_method,
                     skipconnections,
                     min_std_err,
                     loss_weights_refine,
                     truth_uncertain,
                     output_ndims,
                     direction_obs,
                     regularization_L1_beta, regularization_L2_beta,
                     laplacian_penalty, laplacian_error_penalty)

    Flux.loadmodel!(model,model_state)
    model = model |> device
    return apply_model
    #return model
end

"""
    reconstruct(Atype,data_all,fnames_rec;...)

Train a neural network to reconstruct missing data using the training data set
and periodically run the neural network on the test dataset. The data is assumed to be
available on a regular longitude/latitude grid (which is the case of L3
satellite data).

## Mandatory parameters

* `Atype`: array type to use
* `data_all`: list of named tuples. Every tuple should have `filename` and `varname`. `data_all[1]` will be used for training (and perturbed to prevent overfitting). All others entries `data_all[2:end]` will be reconstructed using the training network
at the epochs defined by `save_epochs`.
* `fnames_rec`: vector of filenames corresponding to the entries `data_all[2:end]`


## Optional parameters:

 * `epochs`: the number of epochs (default `1000`)
 * `batch_size`: the size of a mini-batch (default `50`)
 * `enc_nfilter_internal`: number of filters of the internal encoding layers (default `[16,24,36,54]`)
 * `skipconnections`: list of layers with skip connections (default `2:(length(enc_nfilter_internal)+1)`)
 * `clip_grad`: maximum allowed gradient. Elements of the gradients larger than this values will be clipped (default `5.0`).
 * `regularization_L2_beta`: Parameter for L2 regularization (default `0`, i.e. no regularization)
 * `save_epochs`: list of epochs where the results should be saved (default `200:10:epochs`)
 * `is3D`: Switch to apply 2D (`is3D == false`) or 3D (`is3D == true`) convolutions (default `false`)
 * `upsampling_method`: interpolation method during upsampling which can be either `:nearest` or `:bilinear` (default `:nearest`)
 * `ntime_win`: number of time instances within the time window. This number should be odd. (default `3`)
 * `learning_rate`: initial learning rate of the ADAM optimizer (default `0.001`)
 * `learning_rate_decay_epoch`: the exponential decay rate of the learning rate. After `learning_rate_decay_epoch` the learning rate is halved. The learning rate is computed as  `learning_rate * 0.5^(epoch / learning_rate_decay_epoch)`. `learning_rate_decay_epoch` can be `Inf` for a constant learning rate (default)
 * `min_std_err`: minimum error standard deviation preventing a division close to zero (default `exp(-5) = 0.006737946999085467`)
 * `loss_weights_refine`: the weigh of the individual refinement layers using in the cost function. If `loss_weights_refine` has a single element, then there is no refinement.  (default `(1.,)`)
 * `modeldir`: path of the directory to save the model checkpoints.

!!! note
    Note that also the optional parameters should be to tuned for a particular
    application.


Internally the time mean is removed (per default) from the data before it is reconstructed.
The time mean is also added back when the file is saved.
However, the mean is undefined for for are pixels in the data defined as valid (sea) by the mask which do not have any valid data in the training dataset.

See `DINCAE.load_gridded_nc` for more information about the netCDF file.
"""
function reconstruct(Atype,data_all,fnames_rec;
                     epochs = 1000,
                     batch_size = 50,
                     #dropout_rate_train = 0.3,
                     truth_uncertain = false,
                     enc_nfilter_internal = [16,24,36,54],
                     skipconnections = 2:(length(enc_nfilter_internal)+1),
                     clip_grad = 5.0,
                     regularization_L1_beta = 0,
                     regularization_L2_beta = 0,
                     save_epochs = min(epochs,200):10:epochs,
                     is3D = false,
                     upsampling_method = :nearest,
                     ntime_win = 3,
                     learning_rate = 0.001,
                     learning_rate_decay_epoch = Inf,
                     min_std_err = 0.006737946999085467,
                     loss_weights_refine = (1.,),
                     cycle_periods = (365.25,), # days
                     output_ndims = 1,
                     direction_obs = nothing,
                     remove_mean = true,
                     paramfile = nothing,
                     laplacian_penalty = 0,
                     laplacian_error_penalty = laplacian_penalty,
                     modeldir = nothing,
)
    DB(Atype,d,batch_size) = (Atype.(tmp) for tmp in DataLoader(d,batch_size))

    if isempty(save_epochs) || epochs < minimum(save_epochs)
        error("No output will be saved. Consider to adjust save_epochs (currently $save_epochs) or epochs (currently $epochs).")
    end

    device = _to_device(Atype)
    varname = data_all[1][1].varname

    @info "Number of threads: $(Threads.nthreads())"

    train_data = DINCAE.NCData(
        data_all[1],
        train = true,
        ntime_win = ntime_win,
        is3D = is3D,
        cycle_periods = cycle_periods,
        remove_mean = remove_mean,
        direction_obs = direction_obs,
    )

    train_mean_data = train_data.meandata
    # first element in data_all is for training
    data_sources = [train_data, [DINCAE.NCData(
        d,
        train = false,
        ntime_win = ntime_win,
        is3D = is3D,
        cycle_periods = cycle_periods,
        remove_mean = remove_mean,
        # use common mean
        mean_data = train_mean_data,
        direction_obs = direction_obs,
    ) for d in data_all[2:end]]...]

    #data_iter = [PrefetchDataIter(DINCAE.DataBatches(Atype,ds,batch_size)) for ds in data_sources]
    data_iter = [DINCAE.DataBatches(Atype,ds,batch_size) for ds in data_sources]
    #data_iter = [DB(Atype,ds,batch_size) for ds in data_sources]

    train = data_iter[1]
    # try to overfir a single minibatch
    #train = [first(data_iter[1])]

    all_varnames = map(v -> v.varname,data_all[1])

    if output_ndims == 1
        output_varnames = all_varnames[train_data.isoutput]
        # number of output variables
        noutput = sum(train_data.isoutput)
    else
        @assert output_ndims == 2
        output_varnames = ["u","v"]

        noutput = output_ndims
        direction_obs = device(direction_obs)
    end

    @info "Output variables:  $output_varnames"

    inputs_,xtrue = first(train)

    @debug begin
        @info "save first minibatch"
        #JLD2.save("minibatch1.jld2",Dict("inputs" => Array(inputs_),
        #                                 "xtrue" => Array(xtrue)))
    end

    sz = size(inputs_)

    @info "Input size:        $(format_size(sz))"
    @info "Input sum:         $(sum(inputs_))"


    model = genmodel(sz[1:end-2],sz[end-1],noutput;
                     enc_nfilter_internal,
                     upsampling_method,
                     skipconnections,
                     min_std_err,
                     loss_weights_refine,
                     truth_uncertain,
                     output_ndims,
                     direction_obs,
                     regularization_L1_beta, regularization_L2_beta,
                     laplacian_penalty, laplacian_error_penalty)

    @info "using device:      $device"
    model = model |> device

    xrec = model(device(inputs_))
    @info "Output size:       $(format_size(size(xrec)))"
    @info "Output range:      $(extrema(Array(xrec)))"
    @info "Output sum:        $(sum(xrec))"

    #loss = model(device(inputs_), device(xtrue))
    loss = loss_function(model,device(inputs_), device(xtrue))
    @info "Initial loss:      $loss"

    losses = typeof(loss)[]

    for fname_rec in fnames_rec
        if isfile(fname_rec)
            rm(fname_rec)
        end
    end

    ds = [ncsetup(
        fname_rec,output_varnames,(train_data.lon,train_data.lat),
        train_data.meandata[:,:,findall(train_data.isoutput)],
        output_ndims = output_ndims,
    )
          for fname_rec in fnames_rec]

    MO = train_init(model,:ADAM; clip_grad = clip_grad, learning_rate = learning_rate)

    # loop over epochs
    @time for e = 1:epochs
    #Juno.@profile @time  for e = 1:epochs
        lr = learning_rate * 0.5^(e / learning_rate_decay_epoch)

        loss_avg = train_epoch!(MO,DINCAE.PrefetchDataIter(train),lr;
                                clip_grad = clip_grad)

        push!(losses,loss_avg)
        println("epoch: $(@sprintf("%5d",e )) loss $(@sprintf("%5.4f",losses[end]))")
        flush(stdout)

        if e ∈ save_epochs
            if !isnothing(modeldir)
                println("Save model $e")
                model_fname = joinpath(modeldir,"model-checkpoint-" * @sprintf("%05d",e) * ".jld2")
                model_state = Flux.state(cpu(model));
                jldsave(model_fname; model_state,
                        ntime_win, is3D, cycle_periods, remove_mean,
                        train_mean_data,
                        direction_obs, e,
                        noutput,
                        enc_nfilter_internal,
                        upsampling_method,
                        skipconnections,
                        min_std_err,
                        loss_weights_refine,
                        truth_uncertain,
                        output_ndims,
                        regularization_L1_beta, regularization_L2_beta,
                        laplacian_penalty, laplacian_error_penalty,
                        input_size = sz[1:end-1],
                        )
            end

            println("Save output $e")

            for (d_iter,ds_) in zip(data_iter[2:end],ds)
                @time for (ii,(inputs_,xtrue)) in enumerate(d_iter)
                    xrec = Array(model(inputs_))

                    if is3D
                        # take middle frame
                        xrec = xrec[:,:,(end+1)÷2,:,:]
                    end

                    offset = (ii-1)*batch_size

                    savesample(
                        ds_,
                        output_varnames,xrec,
                        train_data.meandata[:,:,findall(train_data.isoutput)],
                        ii-1,offset,
                        output_ndims = output_ndims,
                        mask = train_data.mask,
                    )
                end
            end

            # reduce the risk for memory leaks
            GC.gc()
        end

        shuffle!(train)
    end

    close.(ds)

    if paramfile !== nothing
        NCDataset(paramfile,"c") do ds_
            defVar(ds_,"losses",losses,("epochs",))
            ds_.attrib["epochs"] = epochs
            ds_.attrib["batch_size"] = batch_size
            ds_.attrib["truth_uncertain"] = Int(truth_uncertain)
            ds_.attrib["enc_nfilter_internal"] = Vector{Int}(collect(enc_nfilter_internal))
            ds_.attrib["skipconnections"] = Vector{Int}(collect(skipconnections))
            ds_.attrib["clip_grad"] = clip_grad
            ds_.attrib["regularization_L1_beta"] = regularization_L1_beta
            ds_.attrib["regularization_L2_beta"] = regularization_L2_beta
            ds_.attrib["save_epochs"] = Vector{Int}(save_epochs)
            ds_.attrib["is3D"] = Int(is3D)
            ds_.attrib["upsampling_method"] = string(upsampling_method)
            ds_.attrib["ntime_win"] = ntime_win
            ds_.attrib["learning_rate"] = learning_rate
            ds_.attrib["learning_rate_decay_epoch"] = learning_rate_decay_epoch
            ds_.attrib["min_std_err"] = min_std_err
            ds_.attrib["loss_weights_refine"] = Vector{Float64}(collect(loss_weights_refine))
            ds_.attrib["cycle_periods"] = Vector{Float64}(collect(cycle_periods))
        end
    end

    return losses
end
