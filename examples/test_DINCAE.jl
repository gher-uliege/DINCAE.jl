using DINCAE
using Base.Iterators
using Knet
using Dates
using Printf
using Random
using dincae_utils
using NCDatasets
using Glob

const F = Float32

if Knet.gpu() == -1
    Atype = Array{F}
else
    Atype = KnetArray{F}
end

if Atype == Array{F}
    Knet.gpu(false)
else
    Knet.gpu(true)
end

data = [
   (filename = expanduser("~/tmp/Data/Med/AVHRR/Data/avhrr_sub_add_clouds.nc"),
    varname = "SST",
    obs_err_std = 1,
    jitter_std = 0.05,
    isoutput = true,
   )
]
data_test = data;

outdir = expanduser("~/tmp/Data/DINCAE.jl/Test-simple-new-cat-skip-fixtimeindex-truth_certain_ntime_win3-nearest-wider-rerun-pf-t6-with-refine")

fnames_rec = [joinpath(outdir,"data-avg.nc")]


#------------------


@show outdir
data_all = [data,data_test]

varname = data_all[1][1].varname

mkpath(outdir)

cd(joinpath(dirname(pathof(DINCAE)),"..")) do
    write("$outdir/DINCAE.commit", read(`git rev-parse HEAD`))
    write("$outdir/DINCAE.diff", read(`git diff`))
end;

epochs = 1000
batch_size = 50
save_each = 10
save_model_each = 500
skipconnections = [1,2,3,4]
dropout_rate_train = 0.3
truth_uncertain = false
enc_nfilter_internal = round.(Int,32 * 2 .^ (0:4))


clip_grad = 5.0
regularization_L2_beta = 0
save_epochs = 200:10:epochs
is3D = false


ntime_win = 3
upsampling_method = :nearest

loss_weights_refine = (1.,)
loss_weights_refine = (0.3,0.7)


DINCAE.reconstruct(Atype,data_all,fnames_rec;
                   epochs = epochs,
                   batch_size = batch_size,
                   truth_uncertain = truth_uncertain,
                   enc_nfilter_internal = enc_nfilter_internal,
                   clip_grad = clip_grad,
                   save_epochs = save_epochs,
                   is3D = is3D,
                   upsampling_method = upsampling_method,
                   loss_weights_refine = loss_weights_refine,
                   ntime_win = ntime_win,
)


fnameavg = fnames_rec[1]
@show dincae_utils.cvrms(dincae_utils.AVHRR_case,fnameavg)

