
format_size(sz) = join(string.(sz),"×")

"""
    lon,lat,time,data,missingmask,mask = load_gridded_nc(fname,varname; minfrac = 0.05)

Load the variable `varname` from the NetCDF file `fname`. The variable `lon` is
the longitude in degrees east, `lat` is the latitude in degrees North, `time` is
a numpy datetime vector, `data_full` is a 3-d array with the data, `missingmask` is a boolean mask where true means the data is missing and `mask` is a boolean mask
where true means the data location is valid, e.g. sea points for sea surface temperature.

At the bare-minimum a NetCDF file should have the following variables and
attributes:


    netcdf file.nc {
    dimensions:
            time = UNLIMITED ; // (5266 currently)
            lat = 112 ;
            lon = 112 ;
    variables:
            double lon(lon) ;
            double lat(lat) ;
            double time(time) ;
                    time:units = "days since 1900-01-01 00:00:00" ;
            int mask(lat, lon) ;
            float SST(time, lat, lon) ;
                    SST:_FillValue = -9999.f ;
    }

"""
function load_gridded_nc(fname::AbstractString,varname::AbstractString; minfrac = 0.05)
    ds = Dataset(fname);
    lon = nomissing(ds["lon"][:])
    lat = nomissing(ds["lat"][:])
    time = nomissing(ds["time"][:])
    data = nomissing(ds[varname][:,:,:],NaN)

    #time = nomissing(ds["time"][1:100])
    #data = nomissing(ds[varname][:,:,1:100],NaN)

    if "mask" in ds
        mask = nomissing(ds["mask"][:,:]) .== 1;
    else
        @info("compute mask from $varname: sea point should have at least " *
              "$minfrac for valid data tought time")

        missingfraction = mean(isnan.(data),dims=3)
        println("range of fraction of missing data: ",extrema(missingfraction))
        mask = missingfraction .> minfrac

        println("mask: sea points ",sum(.!mask)," land points ",sum(mask))
    end

    close(ds)
    missingmask = isnan.(data)
    sz = size(data)

    println("$varname data shape: $(format_size(sz)) data range: $(extrema(data[isfinite.(data)]))")

    data4d = reshape(data,(sz[1],sz[2],1,sz[3]))
    return lon,lat,time,data4d,missingmask,mask
end

"""
the first variable is for isoutput is none specified
"""
function load_gridded_nc(data::AbstractVector{NamedTuple{(:filename, :varname, :obs_err_std),T}}) where {T}
    return load_gridded_nc([(d...,isoutput = i==1) for (i,d) in enumerate(data)])
end

function load_gridded_nc(data::AbstractVector{NamedTuple{(:filename, :varname, :obs_err_std, :jitter_std, :isoutput),T}}) where {T}
    lon,lat,datatime,data_full1,missingmask,mask = load_gridded_nc(data[1].filename,data[1].varname);
    sz = size(data_full1)
    data_full = zeros(Float32,sz[1],sz[2],length(data),sz[4]);
    data_full[:,:,1,:] = data_full1;

    for i in 2:length(data)
         lon_,lat_,datatime_,data_full[:,:,i,:],missingmask_,mask_ = load_gridded_nc(
           data[i].filename,data[i].varname);
    end

    return lon,lat,datatime,data_full,missingmask,mask
end

function normalize2(data)
    s = std(data)
    m = mean(data)
    if s == 0
        return data .- m
    else
        return (data .- m) ./ s
    end
end

function load_aux_data(T,sz,auxdata_files)
    auxdata = zeros(T,sz[1:end-1]...,2*length(auxdata_files),sz[end])

    for i = 1:length(auxdata_files)
        NCDataset(auxdata_files[i].filename) do ds

            data = nomissing(Array(ds[auxdata_files[i].varname]))
            data_std_err = nomissing(Array(ds[auxdata_files[i].errvarname]))

            @show "remove time mean from $(auxdata_files[i].filename)"
            data = data .- mapslices(mean,data,dims=[1,2])

            auxdata[:,:,2*i-1,:] = normalize2(data ./ (data_std_err.^2))
            auxdata[:,:,2*i  ,:] = normalize2(1 ./ (data_std_err.^2))
        end
    end

    return auxdata
end


#mutable struct NCData{T,N} <: AbstractVector{Tuple{Array{T,N},Array{T,N}}}
mutable struct NCData{T,N}
    lon::Vector{T}
    lat::Vector{T}
    time::Vector{DateTime}
    data_full::Array{T,4}
    missingmask::BitArray{3}
    meandata::Array{T,3}
    x::Array{T,5}
    isoutput::Vector{Bool}
    train::Bool
    obs_err_std::Vector{T}
    jitter_std::Vector{T}
    lon_scaled::Vector{T}
    lat_scaled::Vector{T}
    dayofyear_cos::Vector{T}
    dayofyear_sin::Vector{T}
    ntime_win::Int
end


Base.length(dd::NCData) = length(dd.time)
size_(dd::NCData) = (length(dd.time),);

@inline function sizex(dd::NCData{T,3}) where T
    # number of parameters
    sz = size(dd.data_full)
    ndata = sz[3]
    ntime_win = dd.ntime_win

    nvar = 4 + ndata*2*ntime_win
    #nvar = 6+2*2
    return (sz[1],sz[2],nvar)
end

@inline function sizey(dd::NCData{T,3}) where T
    sz = size(dd.data_full)
    nout = 2*sum(dd.isoutput)
    return (sz[1],sz[2],nout)
end

@inline function sizex(dd::NCData{T,4}) where T
    # number of parameters
    sz = size(dd.data_full)
    ndata = sz[3]
    ntime_win = dd.ntime_win

    nvar = 4 + ndata*2
    #nvar = 6+2*2
    return (sz[1],sz[2],ntime_win,nvar)
end
export sizex

@inline function sizey(dd::NCData{T,4}) where T
    ntime_win = dd.ntime_win
    sz = size(dd.data_full)
    nout = 2

    return (sz[1],sz[2],ntime_win,2)
end
export sizey


"""
    dd = NCData(lon,lat,time,data_full,missingmask;
                train = false,
                obs_err_std = 1.,
                jitter_std = 0.05)

Return a structure holding the data for training (`train = true`) or testing (`train = false`)
the neural network. `obs_err_std` is the error standard deviation of the
observations. The variable `lon` is the longitude in degrees east, `lat` is the
latitude in degrees North, `time` is a numpy datetime vector, `data_full` is a
3-d array with the data and `missingmask` is a boolean mask where true means the data is
missing. `jitter_std` is the standard deviation of the noise to be added to the
data during training.
"""
function NCData(lon,lat,time,data_full,missingmask;
                train = false,
                obs_err_std = fill(1.,size(data_full,3)),
                jitter_std = fill(0.05,size(data_full,3)),
                ntime_win = 3,
                is3D = false,
                isoutput = (1:size(data_full,3) .== 1),
                )

    meandata = sum(x -> (isnan(x) ? zero(x) : x),data_full,dims = 4) ./ sum(.!isnan,data_full,dims = 4)

    # number of parameters
    ndata = size(data_full,3)
    ntime = size(data_full,4)

    year_length = 365.25 # days
    dayofyear = Dates.dayofyear.(time)
    dayofyear_cos = Float32.(cos.(2π * dayofyear/year_length))
    dayofyear_sin = Float32.(sin.(2π * dayofyear/year_length))

    data = data_full .- meandata
    sz = size(data)

    # dimensions of x: lon, lat, parameter, time, 2
    x = zeros(Float32,(sz[1],sz[2],sz[3],sz[4],2))

    x[:,:,:,:,1] = replace(data,NaN => 0)
    x[:,:,:,:,2] = (1 .- isnan.(data))

    for i = 1:ndata
        # inv. error variance
        x[:,:,i,:,1] ./= obs_err_std[i]^2
        x[:,:,i,:,2] ./= obs_err_std[i]^2
    end
    # scale between -1 and 1
    lon_scaled = Float32.(2 * (lon .- minimum(lon)) / (maximum(lon) - minimum(lon)) .- 1)
    lat_scaled = Float32.(2 * (lat .- minimum(lat)) / (maximum(lat) - minimum(lat)) .- 1)

    N = (is3D ? 4 : 3)

    NCData{Float32,N}(Float32.(lon),Float32.(lat),time,data_full,missingmask,meandata[:,:,:,1],x,
           isoutput,
           train,
           Float32.(obs_err_std),
           Float32.(jitter_std),
           lon_scaled,
           lat_scaled,
           dayofyear_cos,
           dayofyear_sin,
           ntime_win
           )
end


getp(x,sym,default) = (hasproperty(x, sym) ? getproperty(x,sym) : default)

function NCData(data; kwargs...)
    lon,lat,datatime,data_full,missingmask,mask = DINCAE.load_gridded_nc(data)

    default_jitter_std = 0.05

    jitter_std = [getp(d,:jitter_std,default_jitter_std) for d in data]

    return DINCAE.NCData(lon,lat,datatime,data_full,missingmask;
                         obs_err_std = [d.obs_err_std for d in data],
                         jitter_std = jitter_std,
                         isoutput = [d.isoutput for d in data],
                         kwargs...)

end

function getxy!(dd::NCData{T,3},ind::Integer,xin,xtrue) where T

    sz = size(dd.data_full)
    ndata = sz[3]
    ntime = sz[4]
    ntime_win = dd.ntime_win # must be odd
    #@show size(xin,3), 4 + ndata*2*ntime_win

    # central time
    centraln = (ntime_win+1) ÷ 2

    ntime_win_half = (ntime_win-1) ÷ 2
    nrange = max.(1, min.(ntime, (-ntime_win_half:ntime_win_half) .+ ind))

    # metadata
    @inbounds for j = 1:sz[2]
        for i = 1:sz[1]
            xin[i,j,1]  = dd.lon_scaled[i]
            xin[i,j,2]  = dd.lat_scaled[j]
            xin[i,j,3]  = dd.dayofyear_cos[ind]
            xin[i,j,4]  = dd.dayofyear_sin[ind]
        end
    end

    # data (current day)
    @inbounds for idata = 1:ndata
        for (localn,n) in enumerate(nrange)
        yield()

            for j = 1:sz[2]
                for i = 1:sz[1]
                    offset = 5 + 2*(localn-1) + (idata-1)*2*ntime_win
                    xin[i,j,offset] = dd.x[i,j,idata,n,1]
                    xin[i,j,offset + 1] = dd.x[i,j,idata,n,2]
                end
            end
        end
    end

    # add missing data during training randomly
    if dd.train
        imask = rand(1:size(dd.missingmask,3))
        yield()

        @inbounds for j = 1:sz[2]
            for i = 1:sz[1]
                if dd.missingmask[i,j,imask]
                    xin[i,j,5 + 2*(centraln-1)] = 0
                    xin[i,j,6 + 2*(centraln-1)] = 0
                end

                # add jitter
                for idata = 1:ndata
                    for (localn,n) in enumerate(nrange)
                        offset = 5 + 2*(localn-1) + (idata-1)*2*ntime_win

                        xin[i,j,offset] += (dd.jitter_std[idata] * randn(T))
                    end

                    #xin[i,j,5 + (idata-1)*2*ntime_win] += dd.jitter_std[idata] * randn(T)
                    #xin[i,j,7 + (idata-1)*2*ntime_win] += dd.jitter_std[idata] * randn(T)
                    #xin[i,j,9 + (idata-1)*2*ntime_win] += dd.jitter_std[idata] * randn(T)
                end
            end
         end
    end

    offset = 1
    @inbounds for idata in findall(dd.isoutput)
        yield()
        for j = 1:sz[2]
            for i = 1:sz[1]
                xtrue[i,j,offset] = dd.x[i,j,idata,ind,1]
                xtrue[i,j,offset+1] = dd.x[i,j,idata,ind,2]
            end
        end
        offset += 2
    end

    return (xin,xtrue)
end


function getxy!(dd::NCData{T,4},ind::Integer,xin::AbstractArray{T2,4},xtrue::AbstractArray{T2,4}) where {T,T2}

    sz = size(dd.data_full)
    ntime = size(dd.data_full,4)
    ndata = sz[3]
    ntime_win = dd.ntime_win
    #@show size(xin,3), 4 + ndata*2*ntime_win
    ntime_win_half = (ntime_win-1) ÷ 2

    nrange = max.(1, min.(ntime, (-ntime_win_half:ntime_win_half) .+ ind))

    # metadata
    @inbounds for (localn,n) in enumerate(nrange)
        yield()
        for j = 1:sz[2]
            for i = 1:sz[1]
                xin[i,j,localn,1]  = dd.lon_scaled[i]
                xin[i,j,localn,2]  = dd.lat_scaled[j]
                xin[i,j,localn,3]  = dd.dayofyear_cos[n]
                xin[i,j,localn,4]  = dd.dayofyear_sin[n]
            end
        end
    end

    # data
    @inbounds for idata = 1:ndata
        yield()
        for (localn,n) in enumerate(nrange)
            for j = 1:sz[2]
                for i = 1:sz[1]
                    xin[i,j,localn,5 + (idata-1)*2]  = dd.x[i,j,idata,n,1]
                    xin[i,j,localn,6 + (idata-1)*2]  = dd.x[i,j,idata,n,2]
                end
            end
        end
    end

    # add missing data during training randomly
    if dd.train
        for (localn,n) in enumerate(nrange)
            yield()
            imask = rand(1:size(dd.missingmask,3))

            @inbounds for j = 1:sz[2]
                for i = 1:sz[1]
                    if dd.missingmask[i,j,imask]
                        xin[i,j,localn,5] = 0
                        xin[i,j,localn,6] = 0
                    end

                    # add jitter
                    for idata = 1:ndata
                        xin[i,j,localn,5 + (idata-1)*2] += dd.jitter_std[idata] * randn(T)
                    end
                end
            end
         end
    end

    @inbounds for (localn,n) in enumerate(nrange)
        yield()
        for j = 1:sz[2]
            for i = 1:sz[1]
                xtrue[i,j,localn,1] = dd.x[i,j,1,n,1]
                xtrue[i,j,localn,2] = dd.x[i,j,1,n,2]
            end
        end
    end

    return (xin,xtrue)
end

import LearnBase: nobs, getobs, getobs!
nobs(dd::NCData) = length(dd.time)
function getobs(dd::NCData{T},index::Int) where T
    data = (zeros(T,sizex(dd)),zeros(T,sizey(dd)))
    return getobs!(dd,data,index)
end

function getobs!(dd::NCData,data,index::Int) where T
    getxy!(dd,index,data[1],data[2])
    return data
end

function savesample(fname,varnames,xrec,meandata,lon,lat,ii,offset)
    fill_value = -9999.

    if !isfile(fname)
        # create file
        #ds = Dataset(fname, "c", format = :netcdf4_classic)
        ds = Dataset(fname, "c", format = :netcdf3_64bit_offset)

        # dimensions
        defDim(ds,"time", Inf)
        defDim(ds,"lon", length(lon))
        defDim(ds,"lat", length(lat))

        # variables
        nc_lon = defVar(ds,"lon", Float32, ("lon",))
        nc_lat = defVar(ds,"lat", Float32, ("lat",))

        for (i,varname) in enumerate(varnames)
            nc_meandata = defVar(
                ds,
                varname * "_mean", Float32, ("lon","lat"),
                fillvalue=fill_value)

            nc_batch_m_rec = defVar(
                ds,
                varname, Float32, ("lon","lat","time"),
                fillvalue=fill_value)

            nc_batch_sigma_rec = defVar(
                ds,
                varname * "_error", Float32, ("lon","lat","time"),
                fillvalue=fill_value)

            nc_meandata[:,:] = replace(meandata[:,:,i], NaN => missing)
        end
        # data
        nc_lon[:] = lon
        nc_lat[:] = lat
        ds.attrib["count"] = 0
    else
        # append to file
        ds = Dataset(fname, "a")
    end

    if offset == 0
        ds.attrib["count"] = ds.attrib["count"] + 1
    end
    count = Int(ds.attrib["count"])

    if count == 1
        sz = (size(xrec,1),size(xrec,2))

        for varname in varnames
            nc_batch_m_rec = ds[varname]
            nc_batch_sigma_rec = ds[varname * "_error"]

            for n in 1:size(xrec,4)
                nc_batch_m_rec.var[:,:,n+offset] = zeros(Float32,sz)
                nc_batch_sigma_rec.var[:,:,n+offset] = zeros(Float32,sz)
            end
        end
    end

    for (ivar,varname) in enumerate(varnames)
        nc_batch_m_rec = ds[varname]
        nc_batch_sigma_rec = ds[varname * "_error"]

        batch_m_rec = xrec[:,:,2*ivar - 1,:]
        batch_σ2_rec = xrec[:,:,2*ivar,:]

        recdata = batch_m_rec .+ meandata[:,:,ivar]
        batch_sigma_rec = sqrt.(batch_σ2_rec)

        batch_sigma_rec[isnan.(recdata)] .= NaN

        for n in 1:size(batch_m_rec,3)
            # add mask
            #nc_batch_m_rec[:,:,n+offset] = replace(recdata[:,:,n], NaN => missing)
            #nc_batch_sigma_rec[:,:,n+offset] = replace(batch_sigma_rec[:,:,n], NaN => missing)

            nc_batch_m_rec.var[:,:,n+offset] =
                replace(((count-1) * nc_batch_m_rec.var[:,:,n+offset] +
                         recdata[:,:,n]) / count, NaN => fill_value)

            nc_batch_sigma_rec.var[:,:,n+offset] =
                replace(((count-1) * nc_batch_sigma_rec.var[:,:,n+offset] +
                         batch_sigma_rec[:,:,n]) / count, NaN => fill_value)
        end
    end

    close(ds)
end



function ncsetup(fname,varname,lon,lat,meandata)
    fill_value = -9999.

    if isfile(fname)
        rm(fname)
    end

    # create file
    #ncformat = :netcdf4_classic
    ncformat = :netcdf3_64bit_offset
    ds = Dataset(fname, "c", format = ncformat)

    # dimensions
    defDim(ds,"time", Inf)
    defDim(ds,"lon", length(lon))
    defDim(ds,"lat", length(lat))

    # variables
    nc_lon = defVar(ds,"lon", Float32, ("lon",))
    nc_lat = defVar(ds,"lat", Float32, ("lat",))
    nc_meandata = defVar(
        ds,
        varname * "_mean", Float32, ("lon","lat"),
        fillvalue=fill_value)

    nc_batch_m_rec = defVar(
        ds,
        varname, Float32, ("lon","lat","time"),
        fillvalue=fill_value)

    nc_batch_sigma_rec = defVar(
        ds,
        varname * "_error", Float32, ("lon","lat","time"),
        fillvalue=fill_value)

    # data
    nc_lon[:] = lon
    nc_lat[:] = lat
    nc_meandata[:,:] = replace(meandata, NaN => missing);
    ds.attrib["count"] = 0

    return ds
end

function ncsavesample(ds,varname,xrec,meandata,ii,offset)
    fill_value = -9999.
    batch_m_rec = xrec[:,:,1,:]
    batch_σ2_rec = xrec[:,:,2,:]

    recdata = batch_m_rec .+ meandata
    batch_sigma_rec = sqrt.(batch_σ2_rec)

    batch_sigma_rec[isnan.(recdata)] .= NaN

    nc_batch_m_rec = ds[varname]
    nc_batch_sigma_rec = ds[varname * "_error"]

    if offset == 0
        ds.attrib["count"] = ds.attrib["count"] + 1
    end
    count = Int(ds.attrib["count"])

    if count == 1
        sz = (size(batch_m_rec,1),size(batch_m_rec,2))
        for n in 1:size(batch_m_rec,3)
            nc_batch_m_rec.var[:,:,n+offset] = zeros(Float32,sz)
            nc_batch_sigma_rec.var[:,:,n+offset] = zeros(Float32,sz)
        end
    end

    for n in 1:size(batch_m_rec,3)
        # add mask
        #nc_batch_m_rec[:,:,n+offset] = replace(recdata[:,:,n], NaN => missing)
        #nc_batch_sigma_rec[:,:,n+offset] = replace(batch_sigma_rec[:,:,n], NaN => missing)

        nc_batch_m_rec.var[:,:,n+offset] =
            replace(((count-1) * nc_batch_m_rec.var[:,:,n+offset] +
                     recdata[:,:,n]) / count, NaN => fill_value)

        nc_batch_sigma_rec.var[:,:,n+offset] =
            replace(((count-1) * nc_batch_sigma_rec.var[:,:,n+offset] +
                     batch_sigma_rec[:,:,n]) / count, NaN => fill_value)
    end
end



struct DataBatches{Atype,T,N,Tdata,Tbatch}
    data::Tdata
    batchsize::Int
    perm::Vector{Int}
    batch::Tbatch
end

function Random.shuffle!(d::DataBatches)
    randperm!(d.perm)
    return d
end

function DataBatches(Atype,data::NCData{T,N},batchsize) where {T,N}
    perm =
        if data.train
            randperm(nobs(data))
        else
            1:(nobs(data))
        end

    index = perm[1]
    #=
    batch = getobs(data,index)

    T = eltype(batch[1])
    N = ndims(batch[1])
    sizex = size(batch[1])
    sizey = size(batch[2])
    =#
    szx = sizex(data)
    szy = sizey(data)

    batch = (zeros(T,(szx...,batchsize)),
             zeros(T,(szy...,batchsize)))

    return DataBatches{Atype,T,N,typeof(data),typeof(batch)}(
        data,batchsize,perm,batch)
end

function Base.iterate(d::DataBatches{Atype,T,N},index = 0) where {Atype,T,N}
    bs = index+1 : min(index + d.batchsize,length(d.data))

    if length(bs) == 0
        return nothing
    end

    inputs_,xtrue = d.batch
    for ibatch = 1:length(bs)
#    Threads.@threads for ibatch = 1:length(bs)
#    ThreadsX.foreach(1:length(bs)) do ibatch
        #@show Threads.threadid(), ibatch
        j = bs[ibatch]
        if N == 3
            getobs!(d.data,((@view inputs_[:,:,:,ibatch]),(@view xtrue[:,:,:,ibatch])),d.perm[j])
        else
            getobs!(d.data,((@view inputs_[:,:,:,:,ibatch]),(@view xtrue[:,:,:,:,ibatch])),d.perm[j])
        end
    end

#    @show mean(inputs_)
#    @show mean(xtrue)

    if length(bs) == d.batchsize
        # full batch
        return ((Atype(inputs_),Atype(xtrue)),bs[end])
    else
        if N == 3
            return ((Atype(@view inputs_[:,:,:,1:length(bs)]),
                     Atype(@view xtrue[:,:,:,1:length(bs)])),bs[end])
        else
            return ((Atype(@view inputs_[:,:,:,:,1:length(bs)]),
                     Atype(@view xtrue[:,:,:,:,1:length(bs)])),bs[end])
        end
    end
end


# Iterator to prefetch data on a separate thread
mutable struct PrefetchDataIter{T}
    iter::T
    task::Union{Task,Nothing}
end

PrefetchDataIter(iter) = PrefetchDataIter(iter,nothing)

function Base.iterate(d::PrefetchDataIter,args...)
    if d.task == nothing
        out = iterate(d.iter,args...)
    else
        out = fetch(d.task)
    end

    if out == nothing
        return nothing
    else
        next,state = out

        d.task = Threads.@spawn iterate(d.iter,state)
        #d.task = @async iterate(d.iter,state)
        return (next,state)
    end
end

@inline function Random.shuffle!(d::PrefetchDataIter)
    Random.shuffle!(d.iter)
    return d
end
