#ENV["CUDA_VISIBLE_DEVICES"]=""

using DINCAE
using NCDatasets
using Random
using Test
using CUDA
using AMDGPU

T = Float32
filename = "subset-sla-train.nc"
varname = "sla";
auxdata_files = [
  (filename = "subset-sst.nc",
   varname = "SST",
   errvarname = "SST_error")]


if !isfile(filename)
    download("https://dox.ulg.ac.be/index.php/s/YlUX7tVyrflAy8g/download",filename)
end

if !isfile(auxdata_files[1].filename)
    download("https://dox.ulg.ac.be/index.php/s/SvMpLivSG4rLA6I/download",auxdata_files[1].filename)
end

Δlon = 0.25
Δlat = 0.25
lonr = -7 : Δlon : 37
latr = 29 : Δlat : 46
grid = (lonr,latr)

clip_grad = 5.0
jitter_std_pos = (0.17145703272237467f0,0.17145703272237467f0)
learning_rate = 0.0004906558111668519
learning_rate_decay_epoch = 50
epochs = 2
save_epochs = [epochs]
savesnapshot = true
nfilter_inc = 25
batch_size = 32
ndepth = 3
ntime_win = 27
learning_rate = 0.000579728
probability_skip_for_training = 0.877079
seed = 12345
upsampling_method = :nearest
start_skip = 2
regularization_L1_beta = 0
regularization_L2_beta = 1e-4
loss_weights_refine = (1.,)
enc_nfilter_internal = [25,50,75]
skipconnections = start_skip:(length(enc_nfilter_internal)+1)

fnames_rec = ["data-avg.nc"]
paramfile = tempname()

Random.seed!(seed)
Atype =
    if CUDA.functional()
        CuArray{T}
    elseif AMDGPU.functional()
        ROCArray{T}
    else
        Array{T}
    end

DINCAE.reconstruct_points(
    T,Atype,filename,varname,grid,fnames_rec;
    learning_rate = learning_rate,
    learning_rate_decay_epoch = learning_rate_decay_epoch,
    epochs = epochs,
    batch_size = batch_size,
    enc_nfilter_internal = enc_nfilter_internal,
    skipconnections = skipconnections,
    clip_grad = clip_grad,
    save_epochs = save_epochs,
    upsampling_method = upsampling_method,
    jitter_std_pos = jitter_std_pos,
    probability_skip_for_training = probability_skip_for_training,
    auxdata_files = auxdata_files,
    ntime_win = ntime_win,
    savesnapshot = savesnapshot,
    regularization_L1_beta = regularization_L1_beta,
    regularization_L2_beta = regularization_L2_beta,
    loss_weights_refine = loss_weights_refine,
    paramfile = paramfile,
)


@test isfile(fnames_rec[1])

NCDataset(fnames_rec[1]) do ds
    @test haskey(ds,varname)
    @test haskey(ds,varname * "_error")
    @test size(ds[varname],1)  == length(lonr)
    @test size(ds[varname],2)  == length(latr)
end

@test isfile(paramfile)

NCDataset(paramfile) do ds
    @test ds.attrib["epochs"] == epochs
end


@test_throws Exception DINCAE.reconstruct_points(
    T,Atype,filename,varname,grid,fnames_rec;
    epochs = 2,
    save_epochs = [],
)
