var documenterSearchIndex = {"docs":
[{"location":"#DINCAE.jl","page":"Home","title":"DINCAE.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to reconstruct missing data in satellite observations. It can work with gridded data (DINCAE.reconstruct) or a clouds of points (DINCAE.reconstruct_points). In the later case, the data can be organized in e.g. tracks (or not).","category":"page"},{"location":"","page":"Home","title":"Home","text":"The code is available at: https://github.com/gher-uliege/DINCAE.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"The method is described in the following articles:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Barth, A., Alvera-Azcárate, A., Ličer, M., & Beckers, J.-M. (2020). DINCAE 1.0: a convolutional neural network with error estimates to reconstruct sea surface temperature satellite observations. Geoscientific Model Development, 13(3), 1609–1622. https://doi.org/10.5194/gmd-13-1609-2020\nBarth, A., Alvera-Azcárate, A., Troupin, C., & Beckers, J.-M. (2022). DINCAE 2.0: multivariate convolutional neural network with error estimates to reconstruct sea surface temperature satellite and altimetry observations. Geoscientific Model Development, 15(5), 2183–2196. https://doi.org/10.5194/gmd-15-2183-2022","category":"page"},{"location":"","page":"Home","title":"Home","text":"The neural network will be trained on the GPU. Note convolutional neural networks can require a lot of GPU memory depending on the domain size. Flux.jl supports NVIDIA GPUs as well as other vendors (see https://fluxml.ai/Flux.jl/stable/gpu/ for details). Training on the CPU can be performed, but it is prohibitively slow.","category":"page"},{"location":"#User-API","page":"Home","title":"User API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In most cases, a user only needs to interact with the function DINCAE.reconstruct or DINCAE.reconstruct_points.","category":"page"},{"location":"","page":"Home","title":"Home","text":"DINCAE.reconstruct\nDINCAE.reconstruct_points","category":"page"},{"location":"#DINCAE.reconstruct","page":"Home","title":"DINCAE.reconstruct","text":"reconstruct(Atype,data_all,fnames_rec;...)\n\nTrain a neural network to reconstruct missing data using the training data set and periodically run the neural network on the test dataset. The data is assumed to be available on a regular longitude/latitude grid (which is the case of L3 satellite data).\n\nMandatory parameters\n\nAtype: array type to use\ndata_all: list of named tuples. Every tuple should have filename and varname. data_all[1] will be used for training (and perturbed to prevent overfitting). All others entries data_all[2:end] will be reconstructed using the training network\n\nat the epochs defined by save_epochs.\n\nfnames_rec: vector of filenames corresponding to the entries data_all[2:end]\n\nOptional parameters:\n\nepochs: the number of epochs (default 1000)\nbatch_size: the size of a mini-batch (default 50)\nenc_nfilter_internal: number of filters of the internal encoding layers (default [16,24,36,54])\nskipconnections: list of layers with skip connections (default 2:(length(enc_nfilter_internal)+1))\nclip_grad: maximum allowed gradient. Elements of the gradients larger than this values will be clipped (default 5.0).\nregularization_L2_beta: Parameter for L2 regularization (default 0, i.e. no regularization)\nsave_epochs: list of epochs where the results should be saved (default 200:10:epochs)\nis3D: Switch to apply 2D (is3D == false) or 3D (is3D == true) convolutions (default false)\nupsampling_method: interpolation method during upsampling which can be either :nearest or :bilinear (default :nearest)\nntime_win: number of time instances within the time window. This number should be odd. (default 3)\nlearning_rate: initial learning rate of the ADAM optimizer (default 0.001)\nlearning_rate_decay_epoch: the exponential decay rate of the learning rate. After learning_rate_decay_epoch the learning rate is halved. The learning rate is computed as  learning_rate * 0.5^(epoch / learning_rate_decay_epoch). learning_rate_decay_epoch can be Inf for a constant learning rate (default)\nmin_std_err: minimum error standard deviation preventing a division close to zero (default exp(-5) = 0.006737946999085467)\nloss_weights_refine: the weigh of the individual refinement layers using in the cost function. If loss_weights_refine has a single element, then there is no refinement.  (default (1.,))\nmodeldir: path of the directory to save the model checkpoints.\n\nnote: Note\nNote that also the optional parameters should be to tuned for a particular application.\n\nInternally the time mean is removed (per default) from the data before it is reconstructed. The time mean is also added back when the file is saved. However, the mean is undefined for for are pixels in the data defined as valid (sea) by the mask which do not have any valid data in the training dataset.\n\nSee DINCAE.load_gridded_nc for more information about the netCDF file.\n\n\n\n\n\n","category":"function"},{"location":"#DINCAE.reconstruct_points","page":"Home","title":"DINCAE.reconstruct_points","text":"DINCAE.reconstruct_points(T,Atype,filename,varname,grid,fnames_rec )\n\nMandatory parameters:\n\nT: Float32 or Float64: float-type used by the neural network\nArray{T}, CuArray{T},...: array-type used by the neural network.\nfilename: NetCDF file in the format described below.\nvarname: name of the primary variable in the NetCDF file.\ngrid: tuple of ranges with the grid in the longitude and latitude direction e.g. (-180:1:180,-90:1:90).\nfnames_rec: NetCDF file names of the reconstruction.\n\nOptional parameters:\n\njitter_std_pos: standard deviation of the noise to be added to the position of the observations (default (5,5))\nauxdata_files: gridded auxiliary data file for a multivariate reconstruction. auxdata_files is an array of named tuples with the fields (filename, the file name of the NetCDF file, varname the NetCDF name of the primary variable and errvarname the NetCDF name of the expected standard deviation error). For example:\nprobability_skip_for_training: For a given time step n, every track from the same time step n will be skipped by this probability during training (default 0.2). This does not affect the tracks from previous (n-1,n-2,..) and following time steps (n+1,n+2,...). The goal of this parameter is to force the neural network to learn to interpolate the data in time.\nparamfile: the path of the file (netCDF) where the parameter values are stored (default: nothing).\n\nFor example, a single entry of auxdata_files could be:\n\nauxdata_files = [\n  (filename = \"big-sst-file.nc\"),\n   varname = \"SST\",\n   errvarname = \"SST_error\")]\n\nThe data in the file should already be interpolated on the targed grid. The file structure of the NetCDF file is described in DINCAE.load_gridded_nc. The fields defined in this file should not have any missing value (see DIVAnd.ufill).\n\nSee DINCAE.reconstruct for other optional parameters.\n\nAn (minimal) example of the NetCDF file is:\n\nnetcdf all-sla.train {\ndimensions:\n\ttime_instances = 9628 ;\n\tobs = 7445528 ;\nvariables:\n\tint64 size(time_instances) ;\n\t\tsize:sample_dimension = \"obs\" ;\n\tdouble dates(time_instances) ;\n\t\tdates:units = \"days since 1900-01-01 00:00:00\" ;\n\tfloat sla(obs) ;\n\tfloat lon(obs) ;\n\tfloat lat(obs) ;\n\tint64 id(obs) ;\n\tdouble dtime(obs) ;\n\t\tdtime:long_name = \"time of measurement\" ;\n\t\tdtime:units = \"days since 1900-01-01 00:00:00\" ;\n}\n\nThe file should contain the variables lon (longitude), lat (latitude), dtime (time of measurement) and id (numeric identifier, only used by post processing scripts) and dates (time instance of the gridded field). The file should be in the contiguous ragged array representation as specified by the CF convention allowing to group data points into \"features\" (e.g. tracks for altimetry). Every feature can also contain a single data point.\n\n\n\n\n\n","category":"function"},{"location":"#Internal-functions","page":"Home","title":"Internal functions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"DINCAE.load_gridded_nc\nDINCAE.NCData","category":"page"},{"location":"#DINCAE.load_gridded_nc","page":"Home","title":"DINCAE.load_gridded_nc","text":"lon,lat,time,data,missingmask,mask = load_gridded_nc(fname,varname; minfrac = 0.05)\n\nLoad the variable varname from the NetCDF file fname. The variable lon is the longitude in degrees east, lat is the latitude in degrees north, time is a DateTime vector, data_full is a 3-d array with the data, missingmask is a boolean mask where true means the data is missing and mask is a boolean mask where true means the data location is valid, e.g. sea points for sea surface temperature.\n\nAt the bare-minimum a NetCDF file should have the following variables and attributes:\n\nnetcdf file.nc {\ndimensions:\n        time = UNLIMITED ; // (5266 currently)\n        lat = 112 ;\n        lon = 112 ;\nvariables:\n        double lon(lon) ;\n        double lat(lat) ;\n        double time(time) ;\n                time:units = \"days since 1900-01-01 00:00:00\" ;\n        int mask(lat, lon) ;\n        float SST(time, lat, lon) ;\n                SST:_FillValue = -9999.f ;\n}\n\nThe the netCDF mask is 0 for invalid (e.g. land for an ocean application) and 1 for pixels (e.g. ocean).\n\n\n\n\n\n","category":"function"},{"location":"#DINCAE.NCData","page":"Home","title":"DINCAE.NCData","text":"dd = NCData(lon,lat,time,data_full,missingmask,ndims;\n            train = false,\n            obs_err_std = fill(1.,size(data_full,3)),\n            jitter_std = fill(0.05,size(data_full,3)),\n            mask = trues(size(data_full)[1:2]),\n\n)\n\nReturn a structure holding the data for training (train = true) or testing (train = false) the neural network. obs_err_std is the error standard deviation of the observations. The variable lon is the longitude in degrees east, lat is the latitude in degrees north, time is a DateTime vector, data_full is a 3-d array with the data and missingmask is a boolean mask where true means the data is missing. jitter_std is the standard deviation of the noise to be added to the data during training.\n\n\n\n\n\n","category":"type"},{"location":"#Reducing-GPU-memory-usage","page":"Home","title":"Reducing GPU memory usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Convolutional neural networks can require \"a lot\" of GPU memory. These parameters can affect GPU memory utilisation:","category":"page"},{"location":"","page":"Home","title":"Home","text":"reduce the mini-batch size\nuse fewer layers (e.g. enc_nfilter_internal = [16,24,36] or [16,24])\nuse less filters (reduce the values of the optional parameter encnfilterinternal)\nuse a smaller domain or a lower resolution","category":"page"},{"location":"#Troubleshooting","page":"Home","title":"Troubleshooting","text":"","category":"section"},{"location":"#Installation-of-cuDNN","page":"Home","title":"Installation of cuDNN","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you get the warning Package cuDNN not found in current path or the error Scalar indexing is disallowed:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using DINCAE\n┌ Warning: Package cuDNN not found in current path.\n│ - Run `import Pkg; Pkg.add(\"cuDNN\")` to install the cuDNN package, then restart julia.\n│ - If cuDNN is not installed, some Flux functionalities will not be available when running on the GPU.","category":"page"},{"location":"","page":"Home","title":"Home","text":"You need to install and load cuDNN before calling a function in DINCAE.jl:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using cuDNN\nusing DINCAE\n# ...","category":"page"},{"location":"#Dependencies-of-DINCAE.jl","page":"Home","title":"Dependencies of DINCAE.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"DINCAE.jl depends on Flux.jl and CUDA.jl, which will automatically be installed. If you have some problems installing these package you might consult the documentation of Flux.jl or CUDA.jl.","category":"page"}]
}
