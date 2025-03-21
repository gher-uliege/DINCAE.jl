
ENV["CUDA_VISIBLE_DEVICES"]=""

using Test
using DINCAE
using Base.Iterators
using Random
using NCDatasets
using AMDGPU
using CUDA
using Flux
using Printf
using JLD2

const F = Float32
Atype =
    if CUDA.functional()
        CuArray{F}
    elseif AMDGPU.functional()
        ROCArray{F}
    else
        Array{F}
    end

Random.seed!(123)

#filename = "avhrr_sub_add_clouds_small.nc"
filename = "avhrr_sub_add_clouds_n10.nc"

if !isfile(filename)
#    download("https://dox.ulg.ac.be/index.php/s/b3DWpYysuw6itOz/download", filename)
    download("https://dox.ulg.ac.be/index.php/s/2yFgNMkpsGumVSM/download", filename)
end


data = [
   (filename = filename,
    varname = "SST",
    obs_err_std = 1,
    jitter_std = 0.05,
    isoutput = true,
   )
]
data_test = data;
data_all = [data,data_test]

epochs = 3
batch_size = 5
save_each = 10
skipconnections = [1,2]
enc_nfilter_internal = round.(Int,32 * 2 .^ (0:3))
clip_grad = 5.0
regularization_L2_beta = 0
save_epochs = [epochs]
is3D = false
ntime_win = 3

#=
(upsampling_method,is3D,truth_uncertain,loss_weights_refine) = (:nearest, false,false, (1.,))
=#

for (upsampling_method,is3D,truth_uncertain,loss_weights_refine) = (
    (:nearest, false,false, (1.,)),
    (:bilinear,false,false, (1.,)),
    (:nearest, true, false, (1.,)),
    (:nearest, false,true,  (1.,)),
    (:nearest, false,false, (0.3,0.7)),
)

    local fnames_rec = [tempname()]
    paramfile = tempname()

    losses = DINCAE.reconstruct(
        Atype,data_all,fnames_rec;
        epochs,
        batch_size,
        truth_uncertain,
        enc_nfilter_internal,
        clip_grad,
        save_epochs,
        is3D,
        upsampling_method,
        ntime_win,
        loss_weights_refine,
        paramfile,
    )

    @test isfile(fnames_rec[1])

    NCDataset(paramfile) do ds
        losses = ds["losses"][:]
        @test length(losses) == epochs
        @show losses[end]
    end
    rm(fnames_rec[1])
end


# load model and replicate reconstruction

(upsampling_method,is3D,truth_uncertain,loss_weights_refine) = (:nearest, false,false, (1.,))

paramfile = tempname()
modeldir = tempname()
mkpath(modeldir)
fnames_rec = [tempname()]

epochs = 3
save_epochs = [2,3]

losses = DINCAE.reconstruct(
    Atype,data_all,fnames_rec;
    epochs,
    batch_size,
    truth_uncertain,
    enc_nfilter_internal,
    clip_grad,
    save_epochs,
    is3D,
    upsampling_method,
    ntime_win,
    loss_weights_refine,
    paramfile,
    modeldir,
)

data = [
   (filename = filename,
    varname = "SST",
    obs_err_std = 1,
    jitter_std = 0.05,
    isoutput = true,
   )
]

model_fname = joinpath(modeldir,"model-checkpoint-" * @sprintf("%05d",save_epochs[1]) * ".jld2")
JLD2.@load(model_fname,train_mean_data,cycle_periods,remove_mean,is3D)

data_source = DINCAE.NCData(
    data; train = false,
    ntime_win,
    is3D,
    cycle_periods,
    remove_mean,
    mean_data = train_mean_data,
)

batch_size = 2
Atype = Array
data_iter = DINCAE.DataBatches(Atype,data_source,batch_size);

T = Float32
sz = size(data_source.data_full);
m_rec = zeros(T,sz[1],sz[2],sz[4]);
sigma_rec = zeros(T,sz[1],sz[2],sz[4]);
device = gpu

for e in save_epochs
    model_fname = joinpath(modeldir,"model-checkpoint-" * @sprintf("%05d",e) * ".jld2")
    model = DINCAE.loadmodel(model_fname; device);
    offset = 0

    for (xin,xtrue) in data_iter
        batch_m_rec, batch_sigma_rec = model(xin)

        for n = 1:size(batch_m_rec,3)
            m_rec[:,:,n+offset] += batch_m_rec[:,:,n]
            sigma_rec[:,:,n+offset] += batch_sigma_rec[:,:,n]
        end

        offset += size(batch_m_rec,3)
    end
end

m_rec /= length(save_epochs)
sigma_rec /= length(save_epochs)

ds = NCDataset(fnames_rec[1]);
ncSST = nomissing(ds["SST"][:,:,:],NaN);
ncSST_error = nomissing(ds["SST_error"][:,:,:],NaN);

m = isfinite.(ncSST)

@test m_rec[m] ≈ ncSST[m]
@test sigma_rec[m] ≈ ncSST_error[m]

# mirlo: 3.645 ms
#@btime first($data_iter);

fnames_rec = [tempname()]
@test_throws Exception DINCAE.reconstruct(
        Atype,data_all,fnames_rec;
        epochs = 2,
        save_epochs = [],
    )
