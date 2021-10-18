using Dates
using DINCAE
using Knet
using LinearAlgebra
using NCDatasets
using Printf
using Random
#using AutoGrad
using DINCAE_altimetry



T = Float32
filename = expanduser("~/tmp/Altimetry/all-sla.train.nc")
maskname = expanduser("~/tmp/Altimetry/mask.nc");
varname = "sla";

batch_size = 32*2;


Δlon = 0.1
Δlat = 0.1

Δlon = 0.25
Δlat = 0.25

lonr = -7 : Δlon : 37
latr = 29 : Δlat : 46

clip_grad = 5.0

epochs = 150
save_epochs = [epochs]
#save_epochs = 1:10:epochs
enc_nfilter_internal = [25,50,75]
upsampling_method = :nearest

probability_skip_for_training = 0.4
probability_skip_for_training = 1.
jitter_std_pos = (0.17145703272237467f0,0.17145703272237467f0)

learning_rate = 0.0004906558111668519

skipconnections = 3:length(enc_nfilter_internal)
skipconnections = 3:(length(enc_nfilter_internal)+1)
learning_rate_decay_epoch = 50
ntime_win = 27

epochs = 150;
seed = 12345


save_epochs = [epochs]
outdir = expanduser("~/tmp/Data/DINCAE.jl/sla/sla-jitter-train-ntime-fixskip-s1-win$(ntime_win)-res$(Δlon)-epochs$(epochs)-$(upsampling_method)-save$(join(string.(save_epochs),"-"))-skip$(join(string.(skipconnections)))-$(join(string.(enc_nfilter_internal)))-jitter_std_pos$(jitter_std_pos[2])-probability_skip_for_training$(probability_skip_for_training)-learning_rate$(learning_rate)-batch_size$(batch_size)-seed$(seed)-rerun")


fnames_rec = [joinpath(outdir,"data-avg.nc")]

#------


grid = (lonr,latr)

#Knet.gpu(true)
Knet.seed!(seed)
#Knet.gpu(false)

Atype =
    if Knet.gpu() == -1
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
    ntime_win = ntime_win,
)

@show cvrms(fnames_rec[1])

