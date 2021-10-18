using PyCall
using JLD2
using Dates
using DINCAE
using Knet
using LinearAlgebra
using NCDatasets
using Printf
using Random
using Statistics
using AutoGrad
using DINCAE_altimetry
using FileIO
using Glob
using CUDA

#AutoGrad.set_gc_function(AutoGrad.default_gc)

skopt = pyimport("skopt")

const CVRMS_ERROR = 9999.


const CASE = get(ENV,"SLURM_JOB_NAME","")
@show CASE

if CASE == "optimSST"
    const basedir = expanduser("~/tmp/Data/DINCAE.jl/sla-optim-SST3")
else
    const basedir = expanduser("~/tmp/Data/DINCAE.jl/sla-optim3")
end

function cvrms(fname_rec)
    varname = "sla"
    filename_dev = expanduser("~/tmp/Altimetry/all-sla.dev.nc")
    fnamesummary_dev = replace(fname_rec,".nc" => ".dev.json")
    maskname = expanduser("~/tmp/Altimetry/mask.nc");

    try
        summary, = DINCAE_altimetry.errstat(filename_dev,fname_rec,varname,maskname; fnamesummary = fnamesummary_dev)
        return summary["cvrms"]
    catch e
        @show e
        return CVRMS_ERROR
    end
end


function func(x)
    ntime_win = Int(x[1])
    ndepth = Int(x[2])
    nfilter_inc = Int(x[3])
    start_skip = Int(x[4])
    upsampling_method = Symbol(x[5])
    jitter_std_posy = x[6]
    probability_skip_for_training = x[7]
    learning_rate = x[8]
    epochs = Int(x[9])
    batch_size = Int(x[10])
    inv_learning_rate_decay_epoch = x[11]
    regularization_L2_beta = x[12]

    # can be Inf
    learning_rate_decay_epoch = 1. / inv_learning_rate_decay_epoch
    @show x,upsampling_method,learning_rate_decay_epoch

    enc_nfilter_internal = nfilter_inc * (1:ndepth)
    skipconnections = start_skip:(length(enc_nfilter_internal)+1)



    T = Float32
    filename = expanduser("~/tmp/Altimetry/all-sla.train.nc")
    varname = "sla";
    #batch_size = 32;

    regularization_L1_beta = 0
    Δlon = 0.25
    Δlat = 0.25

    lonr = -7 : Δlon : 37
    latr = 29 : Δlat : 46

    jitter_std_pos = (Float32(jitter_std_posy / cosd(mean(latr))),
                      Float32(jitter_std_posy))

    clip_grad = 5.0
    save_epochs = [epochs]

    seed = 12345

    auxdata_files = []
    if CASE == "optimSST"
        auxdata_files = [
        (filename = expanduser("~/tmp/Altimetry/all-sst.nc"),
         varname = "SST",
         errvarname = "SST_error")
        ]
    end


    outdir = joinpath(basedir,expanduser("$(Dates.format(Dates.now(),"yyyymmddTHHMMSS"))"))
    skipconnections_fix = true

    @show enc_nfilter_internal
    @show seed
    @show skipconnections
    @show ntime_win
    @show ndepth
    @show nfilter_inc
    @show start_skip
    @show upsampling_method
    @show save_epochs
    @show epochs
    @show clip_grad
    @show batch_size
    @show lonr
    @show latr
    @show jitter_std_posy
    @show probability_skip_for_training
    @show learning_rate
    @show skipconnections_fix
    @show inv_learning_rate_decay_epoch
    @show regularization_L1_beta
    @show regularization_L2_beta

    @show outdir
    mkpath(outdir)
    JLD2.@save joinpath(outdir,"params.jld2") enc_nfilter_internal seed  skipconnections ntime_win ndepth nfilter_inc start_skip upsampling_method save_epochs epochs clip_grad batch_size lonr latr jitter_std_posy probability_skip_for_training learning_rate skipconnections_fix inv_learning_rate_decay_epoch regularization_L2_beta regularization_L1_beta

    fnames_rec = [joinpath(outdir,"data-avg.nc")]

    grid = (lonr,latr)

    #Knet.seed!(seed)
    #CUDA.seed!(seed)
    Atype = KnetArray{T}
    #Atype = (Knet.gpu() == -1 ? Array{T} : KnetArray{T})

    DINCAE.reconstruct_points(
        T,Atype,filename,varname,grid,fnames_rec;
        epochs = epochs,
        batch_size = batch_size,
        enc_nfilter_internal = enc_nfilter_internal,
        skipconnections = skipconnections,
        clip_grad = clip_grad,
        save_epochs = save_epochs,
        upsampling_method = upsampling_method,
        ntime_win = ntime_win,
        jitter_std_pos = jitter_std_pos,
        probability_skip_for_training = probability_skip_for_training,
        learning_rate = learning_rate,
        learning_rate_decay_epoch = learning_rate_decay_epoch,
        auxdata_files = auxdata_files,
        regularization_L1_beta = regularization_L1_beta,
        regularization_L2_beta = regularization_L2_beta,
    )

    sleep(10)
    run(`sync`)
    @show stat(fnames_rec[1]).size

    res = cvrms(fnames_rec[1])
    @show res

    GC.gc()
    Knet.gc()
    CUDA.memory_status()
    CUDA.reclaim()
    CUDA.memory_status()

    #return (x[1]-1)^2 + x[2]^2 + x[3]
    return res
end

function filterparam(x0,y0,dimensions)
    keep = [all([dimensions[j].__contains__(x0[i,j]) for j in 1:length(dimensions)]) for i = 1:size(x0,1)]
    ik = findall(keep .& (y0 .!= CVRMS_ERROR))
    return x0[ik,:],y0[ik]
end


max_nfilter_inc = (get(ENV,"CLUSTER_NAME","") == "dragon2" ? 32 : 25)

dimensions = [
    skopt.space.Integer(low=3, high = 41, name = "ntime_win"),
    skopt.space.Integer(low=3, high = 10, name = "ndepth"),
    skopt.space.Integer(low=5, high = max_nfilter_inc, name = "nfilter_inc"),
    skopt.space.Integer(low=1, high = 10, name = "start_skip"),
    skopt.space.Categorical(["nearest","bilinear"], name = "upsampling_method"),
    skopt.space.Real(low=0, high = 2., name = "jitter_std_posy"),
    skopt.space.Real(low=0, high = 1., name = "probability_skip_for_training"),
    skopt.space.Real(low=1e-6, high = 1e-3, name = "learning_rate"),
    skopt.space.Integer(low=60, high = 1000, name = "epochs"),
    skopt.space.Integer(low=32, high = 64, name = "batch_size"),
    skopt.space.Integer(low=0, high = 1/50, name = "inv_learning_rate_decay_epoch"),
    skopt.space.Real(low=0, high = 1e-2, name = "regularization_L2_beta"),
]


function loadparam(filename)
    j = FileIO.load(replace(filename,"data-avg.nc" => "params.jld2"))

    return [j["ntime_win"],j["ndepth"],j["nfilter_inc"],j["start_skip"],
            String(j["upsampling_method"]),
            j["jitter_std_posy"],
            j["probability_skip_for_training"],
            j["learning_rate"],
            j["epochs"],
            j["batch_size"],
            get(j,"inv_learning_rate_decay_epoch",0.),
            get(j,"regularization_L2_beta",0.),
            ]
end


mkpath(basedir)

filenames = map(d -> joinpath(d,"data-avg.nc"),
                filter(d -> (isdir(d) & (isfile(joinpath(d,"data-avg.nc")) || isfile(joinpath(d,"data-avg.dev.json")))) ,
                       readdir(basedir,join=true)))



mkpath(basedir)

cd(joinpath(dirname(pathof(DINCAE)),"..")) do
    write("$basedir/DINCAE2.commit", read(`git rev-parse HEAD`))
    write("$basedir/DINCAE2.diff", read(`git diff`))
end;


n_calls = 50
n_calls = 500
n_calls = 120
# the number of random initialization points
n_random_starts = 10

if length(filenames) == 0
    @info "cold start"
    x0 = [15,7,20,1,"bilinear",0.2,0.56,1e-4,120,32,0.,0.]
    y0 = nothing
    # test
    #func(x0)

    res = skopt.gp_minimize(func,dimensions,n_calls=n_calls,x0 = PyCall.PyVector(x0));
else
    @info "hot start"
    y0 = cvrms.(filenames)
    x0 = copy(permutedims(reduce(hcat,loadparam.(filenames))))

    x0,y0 = filterparam(x0,y0,dimensions)

    display([x0 y0])

    @show max(10 - length(y0),0)
    res = skopt.gp_minimize(
        func,
        dimensions,
        n_calls = n_calls,
        n_random_starts = max(n_random_starts - length(y0),0),
        x0 = [PyVector(x0[i,:]) for i = 1:size(x0,1)],
        y0 = y0);
end

@show res

x_iters = res["x_iters"]
func_vals = res["func_vals"]
fun = res["fun"]
x = res["x"]

@show (x,x_iters,func_vals,fun)

@save joinpath(basedir,"DINCAE_altimetry_optim_res.jld2") x x_iters func_vals fun
