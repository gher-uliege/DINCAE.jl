
ENV["CUDA_VISIBLE_DEVICES"]=""

using Test
using DINCAE
using Base.Iterators
using Random

const F = Float32
Atype = Array{F}

Random.seed!(123)

filename = "avhrr_sub_add_clouds_small.nc"

if !isfile(filename)
    download("https://dox.ulg.ac.be/index.php/s/b3DWpYysuw6itOz/download", filename)
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

fnames_rec = [tempname()]

data_all = [data,data_test]

epochs = 2
batch_size = 50
save_each = 10
save_model_each = 500
skipconnections = [1,2,3,4]
dropout_rate_train = 0.3
truth_uncertain = false
enc_nfilter_internal = round.(Int,32 * 2 .^ (0:4))
clip_grad = 5.0
regularization_L2_beta = 0
save_epochs = []
is3D = false
ntime_win = 3
upsampling_method = :nearest

losses = DINCAE.reconstruct(
    Atype,data_all,fnames_rec;
    epochs = epochs,
    batch_size = batch_size,
    truth_uncertain = truth_uncertain,
    enc_nfilter_internal = enc_nfilter_internal,
    clip_grad = clip_grad,
    save_epochs = save_epochs,
    is3D = is3D,
    upsampling_method = upsampling_method,
    ntime_win = ntime_win,
)

if haskey(ENV,"CI")
    # Volks-Wagen
    @test losses[end] ≈ 23.8846822232148 rtol=1e-4
    @show losses[end]
else
    @test losses[end] == 23.8846822232148
end


