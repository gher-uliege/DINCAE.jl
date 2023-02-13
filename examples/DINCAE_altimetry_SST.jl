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
#filename = expanduser("~/tmp/Altimetry/all-sla.nc")
varname = "sla";

maskname = expanduser("~/tmp/Altimetry/mask.nc");
basedir = expanduser("~/tmp/Data/DINCAE.jl/sla/")

auxdata_files = [
  (filename = expanduser("~/tmp/Altimetry/all-sst.nc"),
   varname = "SST",
   errvarname = "SST_error")]

batch_size = 32*2;


Δlon = 0.1
Δlat = 0.1

Δlon = 0.25
Δlat = 0.25

#Δlon = 1
#Δlat = 1

lonr = -7 : Δlon : 37
latr = 29 : Δlat : 46



clip_grad = 5.0# ,
epochs = 10
epochs = 1
epochs = 500
save_epochs = 60:10:epochs

epochs = 150
#epochs = 1
#epochs = 120
#epochs = 500
#epochs = 1000
#save_epochs = 30:10:epochs
save_epochs = [epochs]
#save_epochs = 1:10:epochs

#sla-jitter-train-ntime-win11-res0.25-epochs60-nearest-save60-skip234567-20406080100120140-seed12345/data-avg.dev.json

enc_nfilter_internal = [10,20,30,40,50,60]
#enc_nfilter_internal = [10,20,30,40,50]
#enc_nfilter_internal = [10,20,30,40]
#enc_nfilter_internal = [32,32,64,64,72,72,94]

enc_nfilter_internal = [20,40,60,80,100,120,140]
enc_nfilter_internal = [25,50,75]

#enc_nfilter_internal = round.(Int,16 * 1.9 .^ (0:4))

upsampling_method = :nearest
#upsampling_method = :bilinear

probability_skip_for_training = 0.4
probability_skip_for_training = 1.
jitter_std_pos = (5f0,5f0)
jitter_std_pos = (2.f0,2.f0)
jitter_std_pos = (0.f0,0.f0)
jitter_std_pos = (0.17145703272237467f0,0.17145703272237467f0)



#for ntime_win = [7,9,5,11]
ntime_win = 11

ntime_win = 21
learning_rate = 0.0004906558111668519

skipconnections = 3:length(enc_nfilter_internal) # cvrms 0.03677049f0
skipconnections = 3:(length(enc_nfilter_internal)+1) # cvrms 0.03689562f0
learning_rate_decay_epoch = 50 # 0.036640305f0
ntime_win = 27 # 0.036808904f0
epochs = 150;

# RMS 0.03737591
#with SST
#cvrms(fnames_rec[1]) 0.0449484

#without SST 0.035873376f0
#without SST without coord and time 0.03780291f0
#auxdata_files = []
# StepModel
#epochs = 150;
#save_epochs = [epochs]



#epochs = 250
#save_epochs = [epochs]
# RMS 0.036021363
epochs = 1000
save_epochs = 0:25:1000
#save_epochs = 0:4
savesnapshot = true
skipconnections = 2:4
nfilter_inc = 25
clip_grad = 5.0
inv_learning_rate_decay_epoch = 0
batch_size = 32
ndepth = 3
enc_nfilter_internal = 25:25:75
#enc_nfilter_internal = [25:25:75; 75]
#enc_nfilter_internal = 25:25:100
enc_nfilter_internal = 25:25:125
#enc_nfilter_internal = 25:25:125
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
#loss_weights_refine = (0.3,0.7)
#loss_weights_refine = (0.2,0.1,0.7)

skipconnections = start_skip:(length(enc_nfilter_internal)+1)


seed = 12345


#outdir = expanduser("~/tmp/Data/DINCAE.jl/sla/sla-jitter-train-ntime-fixskip-s1-win$(ntime_win)-res$(Δlon)-epochs$(epochs)-$(upsampling_method)-save$(join(string.(save_epochs),"-"))-skip$(join(string.(skipconnections)))-$(join(string.(enc_nfilter_internal)))-jitter_std_pos$(jitter_std_pos[2])-probability_skip_for_training$(probability_skip_for_training)-learning_rate$(learning_rate)-batch_size$(batch_size)-seed$(seed)-SST-snapshot")

#outdir = expanduser("~/tmp/Data/DINCAE.jl/sla/sla-jitter-train-ntime-win$(ntime_win)-res$(Δlon)-epochs$(epochs)-$(upsampling_method)-last-skip$(join(string.(skipconnections)))-$(join(string.(enc_nfilter_internal)))-seed$(seed)")
outdir = joinpath(basedir,expanduser("$(Dates.format(Dates.now(),"yyyymmddTHHMMSS"))"))

outdir = joinpath(basedir,expanduser("20201012T115902-rerun-refine"))
outdir = joinpath(basedir,expanduser("20201012T115902-rerun-SSTa"))

@show outdir
fnames_rec = [joinpath(outdir,"data-avg.nc")]

#------


grid = (lonr,latr)

#Knet.gpu(true)
Random.seed!(seed)
CUDA.seed!(Random.rand(UInt64))
#Knet.gpu(false)

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
