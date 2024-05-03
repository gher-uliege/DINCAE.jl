using DINCAE

const F = Float32
Atype = Array{F}

filename = "avhrr_sub_add_clouds_n10.nc"

if !isfile(filename)
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

fnames_rec = [tempname()]
paramfile = tempname()

losses = DINCAE.reconstruct(
    Atype,data_all,fnames_rec;
    epochs = 3,
    batch_size = 5,
    enc_nfilter_internal = round.(Int,32 * 2 .^ (0:3)),
    clip_grad = 5.0,
    upsampling_method = :nearest,
    ntime_win = 3,
    paramfile = paramfile,
    regularization_L2_beta = 0.001,
    loss_weights_refine = (0.3,0.7),
)
