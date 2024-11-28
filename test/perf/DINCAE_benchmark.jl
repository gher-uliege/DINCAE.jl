# # Benchmark DINCAE
#
# Load the necessary modules

using CUDA
using DINCAE
using Dates
using NCDatasets
using AMDGPU

# ## Data download
#
# First we define the domain and time interval.
#
# The example is tested here for a short time frame but for realistic
# applications one should use a much longer time-range (like 10, 20 years or more)

# local directory
localdir = expanduser("~/Data/SST-AlboranSea-example")
# create directory
mkpath(localdir)
# filename of the subset
fname_subset = joinpath(localdir,"modis_subset.nc")
# filename of the clean data
fname = joinpath(localdir,"modis_cleanup.nc")
# filename of the data with added clouds for cross-validation
fname_cv = joinpath(localdir,"modis_cleanup_add_clouds.nc")
varname = "sst"

# Results of DINCAE will be placed in a sub-directory under `localdir`

outdir = joinpath(localdir,"Results")
mkpath(outdir)


# # Reconstruct missing data
#
#
# F is the floating point number type for the neural network. Here we use
# single precision.

const F = Float32

# Test if CUDA is functional to use the GPU, otherwise the CPU is used.

if CUDA.functional()
    Atype = CuArray{F}
elseif AMDGPU.functional()
    Atype = ROMArray{F}
else
    @warn "No supported GPU found. We will use the CPU which is very slow. Please check https://developer.nvidia.com/cuda-gpus"
    Atype = Array{F}
end

# Setting the parameters of neural network.
# See the documentation of `DINCAE.reconstruct` for more information.

epochs = 100
batch_size = 32
enc_nfilter_internal = round.(Int,32 * 2 .^ (0:4))
clip_grad = 5.0
regularization_L2_beta = 0
ntime_win = 3
upsampling_method = :nearest
loss_weights_refine = (0.3,0.7)
save_epochs = [epochs]


data = [
   (filename = fname_cv,
    varname = varname,
    obs_err_std = 1,
    jitter_std = 0.05,
    isoutput = true,
   )
]
data_test = data;
fnames_rec = [joinpath(outdir,"data-avg-benchmark.nc")]
data_all = [data,data_test]

# Use these parameters for a quick test:

# epochs = 10
# save_epochs = epochs:epochs

# Start the training and reconstruction of the neural network

loss = DINCAE.reconstruct(
    Atype,data_all,fnames_rec;
    epochs = epochs,
    batch_size = batch_size,
    enc_nfilter_internal = enc_nfilter_internal,
    clip_grad = clip_grad,
    save_epochs = save_epochs,
    upsampling_method = upsampling_method,
    loss_weights_refine = loss_weights_refine,
    ntime_win = ntime_win,
)

@show loss[1]

# # Post process results
#
# Compute the RMS (Root Mean Squared error) with the independent validation data

#=
case = (
    fname_orig = fname,
    fname_cv = fname_cv,
    varname = varname,
)
fnameavg = joinpath(outdir,"data-avg.nc")
cvrms = DINCAE_utils.cvrms(case,fnameavg)
@info "Cross-validation RMS error is: $cvrms"

# Next we plot all time instances. The figures with be placed in the
# directory `figdir`

figdir = joinpath(outdir,"Fig")
DINCAE_utils.plotres(case,fnameavg, clim = nothing, figdir = figdir,
                     clim_quantile = (0.01,0.99),
                     which_plot = :cv)
@info "Figures are in $(figdir)"


# Example reconstruction for 2001-09-12
# ![reconstruction for the 2001-09-12](Fig/data-avg_2001-09-12.png)
# Panel (a) is the original data where we have added clouds (panel (b)). The
# reconstuction based on the data in panel (b) is shown in panel (c) together
# with its expected standard deviation error (panel (d)).

=#
