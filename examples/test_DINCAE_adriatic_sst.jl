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

Atype = KnetArray{F}
#Atype = Array{F}

if Atype == Array{F}
    Knet.gpu(false)
else
    Knet.gpu(true)
end

data = [
   (filename = expanduser("~/tmp/Data/Med/AVHRR/Data/avhrr_sub_add_clouds.nc"),
    varname = "SST"
   )
]
data_test = data;

outdir = expanduser("~/tmp/Data/DINCAE.jl/Test-simple-odd-test")

fnames_rec = [joinpath(outdir,"data-avg.nc")]

basedir = expanduser("~/Data/DINCAE-multivariate/Adriatic2/")

data_param = [
    (filename = joinpath(basedir,"modis_sst_revlat_add_clouds.nc"),
     varname = "sst_t",
     obs_err_std = 1,
     jitter_std = 0.05f0, # 4.962758f0
     isoutput = true,
     ),
    (filename = joinpath(basedir,"color_revlat_log.nc"),
     varname = "chlor_a",
     obs_err_std = 1,
     jitter_std = 0.01f0 * 0.73579454f0,
     isoutput = false,
     ),
    (filename = joinpath(basedir,"CCMP_Wind_Analysis_Adriatic_revlat_speed_filtered.nc"),
     varname = "wind_speed",
     obs_err_std = 1,
     jitter_std = 0.01f0 * 1.1857846f0,
     isoutput = false,
     ),
    (filename = joinpath(basedir,"CCMP_Wind_Analysis_Adriatic_revlat.nc"),
     varname = "uwnd",
     obs_err_std = 1,
     jitter_std = 0.01f0 * 2.079493f0,
     isoutput = false,
     ),
    (filename = joinpath(basedir,"CCMP_Wind_Analysis_Adriatic_revlat.nc"),
     varname =  "vwnd",
     obs_err_std = 1,
     jitter_std = 0.01f0 * 2.60915f0,
     isoutput = false,
     ),
]

loss_weights_refine = (1.,)
#loss_weights_refine = (0.3,0.7)

for ind in [1:1, 1:2, 1:3, 1:5, [1,3], [1,4,5], [1,3,4,5]]
    data = data_param[ind]

    data_test = data;


    outdir = expanduser("~/tmp/Data/DINCAE.jl/Adriatic2-test-" * join([d.varname for d in data],"-") * "-nfilter1.9-all-3")
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

    enc_nfilter_internal = round.(Int,16 * 1.9 .^ (0:4))

    epochs = 1000
    save_epochs = 200:10:epochs


    mkpath(outdir)

    DINCAE.reconstruct(
        Atype,data_all,fnames_rec;
        epochs = epochs,
        save_epochs = save_epochs,
        enc_nfilter_internal = enc_nfilter_internal,
        loss_weights_refine = loss_weights_refine,
    )

end
