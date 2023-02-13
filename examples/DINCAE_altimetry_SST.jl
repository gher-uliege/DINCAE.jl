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

#=

netcdf all-sla.train {
dimensions:
	track = 9628 ;
	time = 7453066 ;
variables:
	int64 size(track) ;
		size:sample_dimension = "time" ;
	double dates(track) ;
		dates:units = "days since 1900-01-01 00:00:00" ;
	float sla(time) ;
	float lon(time) ;
	float lat(time) ;
	int64 id(time) ;
	double dtime(time) ;
		dtime:long_name = "time of measurement" ;
		dtime:units = "days since 1900-01-01 00:00:00" ;
}

netcdf all-sst {
dimensions:
	lon = 177 ;
	lat = 69 ;
	time = 9628 ;
variables:
	double lon(lon) ;
	double lat(lat) ;
	double time(time) ;
		time:units = "days since 1900-01-01 00:00:00" ;
	float SST(time, lat, lon) ;
	float SST_error(time, lat, lon) ;
}

netcdf mask {
dimensions:
	lon = 177 ;
	lat = 69 ;
variables:
	byte mask(lat, lon) ;
}


=#

T = Float32

# Training data
filename = expanduser("~/tmp/Altimetry/all-sla.train.nc")

# Name of the NetCDF variables
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

# ask me about these parameters
clip_grad = 5.0
jitter_std_pos = (0.17145703272237467f0,0.17145703272237467f0)
learning_rate_decay_epoch = 50
epochs = 1000
save_epochs = 0:25:1000
savesnapshot = true
skipconnections = 2:4
nfilter_inc = 25
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

