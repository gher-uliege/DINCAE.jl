using Dates
using DINCAE
using Knet
using LinearAlgebra
using NCDatasets
using Printf
using Random
#using AutoGrad
using DINCAE_altimetry
using CUDA
using Glob
using Base.Threads

@info "using $(Threads.nthreads()) thread(s)"


T = Float32
filename = expanduser("~/tmp/Altimetry/all-sla.train.nc")
varname = "sla";

maskname = expanduser("~/tmp/Altimetry/mask.nc");
basedir = expanduser("~/tmp/Data/DINCAE.jl/sla/")

auxdata_files = [
  (filename = expanduser("~/tmp/Altimetry/all-sst.nc"),
   varname = "SST",
   errvarname = "SST_error")]

batch_size = 32*2;


Δlon = 0.25
Δlat = 0.25


lonr = -7 : Δlon : 37
latr = 29 : Δlat : 46

clip_grad = 5.0
epochs = 10
epochs = 1
epochs = 500
save_epochs = 60:10:epochs

epochs = 150
save_epochs = [epochs]
enc_nfilter_internal = [25,50,75]

upsampling_method = :nearest

jitter_std_pos = (0.17145703272237467f0,0.17145703272237467f0)


ntime_win = 21
learning_rate = 0.0004906558111668519

skipconnections = 3:(length(enc_nfilter_internal)+1)
learning_rate_decay_epoch = 50
ntime_win = 27
epochs = 150;

epochs = 1000
save_epochs = 0:25:1000
savesnapshot = true
skipconnections = 2:4
nfilter_inc = 25
clip_grad = 5.0
inv_learning_rate_decay_epoch = 0
batch_size = 32
ndepth = 3
enc_nfilter_internal = 32:32:(3*32)

ntime_win = 27
jitter_std_posy = 0.549114
learning_rate = 0.000579728
probability_skip_for_training = 0.877079
seed = 12345
upsampling_method = :nearest
start_skip = 2

regularization_L1_beta = 0
regularization_L2_beta = 1e-4

loss_weights_refine = (1.,)

skipconnections = start_skip:(length(enc_nfilter_internal)+1)


seed = 12345


outdir = joinpath(basedir,expanduser("$(Dates.format(Dates.now(),"yyyymmddTHHMMSS"))"))

@show outdir
fnames_rec = [joinpath(outdir,"data-avg.nc")]

#------


grid = (lonr,latr)

Random.seed!(seed)
CUDA.seed!(Random.rand(UInt64))

Atype =
    if length(CUDA.devices()) == 0
        Array{T}
    else
        KnetArray{T}
    end

mkpath(outdir)

cd(joinpath(dirname(pathof(DINCAE)),"..")) do
    write("$outdir/DINCAE.commit", read(`git rev-parse HEAD`))
    write("$outdir/DINCAE.diff", read(`git diff`))
end;


function cvrms(fname_rec)
    varname = "sla"

    filename_dev = expanduser("~/tmp/Altimetry/all-sla.dev.nc")
    fnamesummary_dev = replace(fname_rec,".nc" => ".dev.json")

    filename_test = expanduser("~/tmp/Altimetry/all-sla.test.nc")
    fnamesummary_test = replace(fname_rec,".nc" => ".test.json")

    summary, = DINCAE_altimetry.errstat(filename_dev,fname_rec,varname,maskname; fnamesummary = fnamesummary_dev)
    return summary["cvrms"]
end

@show regularization_L1_beta
@show regularization_L2_beta

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
)


@show cvrms(fnames_rec[1])
#end



cvrmsn = DINCAE_altimetry.cvrms.(glob("data-avg-epoch*nc",outdir))
@show cvrmsn[end]
