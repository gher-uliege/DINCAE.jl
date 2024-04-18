
format_size(sz) = join(string.(sz),"×")

"""
    lon,lat,time,data,missingmask,mask = load_gridded_nc(fname,varname; minfrac = 0.05)

Load the variable `varname` from the NetCDF file `fname`. The variable `lon` is
the longitude in degrees east, `lat` is the latitude in degrees north, `time` is
a DateTime vector, `data_full` is a 3-d array with the data, `missingmask` is a boolean mask where true means the data is missing and `mask` is a boolean mask
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


The the netCDF mask is 0 for invalid (e.g. land for an ocean application) and 1 for pixels (e.g. ocean).
"""
function load_gridded_nc(fname::AbstractString,varname::AbstractString; minfrac = 0.05)
    ds = Dataset(fname);
    lon = nomissing(ds["lon"][:])
    lat = nomissing(ds["lat"][:])
    time = nomissing(ds["time"][:])
    data = nomissing(ds[varname][:,:,:],NaN)

    #time = nomissing(ds["time"][1:100])
    #data = nomissing(ds[varname][:,:,1:100],NaN)

    if haskey(ds,"mask")
        mask = nomissing(ds["mask"][:,:]) .== 1;
    else
        @info("compute mask from $varname: each sea point should have at least " *
              "$minfrac for valid data through time")

        missingfraction = mean(isnan.(data),dims=3)
        println("range of fraction of missing data: ",extrema(missingfraction))
        mask = missingfraction .<= 1 - minfrac

        println("mask: sea points ",sum(mask)," land points ",sum(.!mask))
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

function load_gridded_nc(data)
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

            data_std_err =
                if isnothing(auxdata_files[i].errvarname)
                    ones(size(data))
                else
                    nomissing(Array(ds[auxdata_files[i].errvarname]))
                end

            @info("remove time mean from $(auxdata_files[i].filename)")
            data = data .- mapslices(mean,data,dims=[1,2])

            auxdata[:,:,2*i-1,:] = normalize2(data ./ (data_std_err.^2))
            auxdata[:,:,2*i  ,:] = normalize2(1 ./ (data_std_err.^2))
        end
    end

    return auxdata
end


#mutable struct NCData{T,N} <: AbstractVector{Tuple{Array{T,N},Array{T,N}}}
mutable struct NCData{T,N #=,TA=#}
    lon::Vector{T}
    lat::Vector{T}
    time::Vector{DateTime}
    data_full::Array{T,4}
    missingmask::BitArray{3}
    meandata::Array{T,3}
    mask::BitMatrix
    x::Array{T,5}
    isoutput::Vector{Bool}
    train::Bool
    obs_err_std::Vector{T}
    jitter_std::Vector{T}
    lon_scaled::Vector{T}
    lat_scaled::Vector{T}
    time_cos::Matrix{T}
    time_sin::Matrix{T}
    ntime_win::Int
    direction_obs::Array{T,3}
    output_ndims::Int
    ndims::Vector{Int}
#    auxdata::TA
end


Base.length(dd::NCData) = length(dd.time)
size_(dd::NCData) = (length(dd.time),);


function nscalar_per_obs_(ndims)
    #  ndims = 1 -> nscalar_per_obs = 1 + 1
    #  ndims = 2 -> nscalar_per_obs = 2 + 2+1
    #  ndims = 3 -> nscalar_per_obs = 3 + 3+2+1

    # mean + lower part of covariance matrix
    return ndims + ndims * (ndims+1) ÷ 2
end


@inline function sizex(dd::NCData{T,3}) where T
    # number of parameters
    sz = size(dd.data_full)
    ndata = sz[3]
    ntime_win = dd.ntime_win

    # if dd.output_ndims == 1
    #     nvar = 4 + ndata*2*ntime_win
    # else
    #     nvar = 4 + ndata*5*ntime_win
    # end

    nvar = 4 + sum(nscalar_per_obs_.(dd.ndims)) * ntime_win
    #nvar = 6+2*2
    return (sz[1],sz[2],nvar)
end


### size of xtrue
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

    nvar = 2 + 2*size(dd.time_cos,1) + ndata*2
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
    dd = NCData(lon,lat,time,data_full,missingmask,ndims;
                train = false,
                obs_err_std = 1.,
                jitter_std = 0.05,
                mask = trues(size(data_full)[1:2]),
)

Return a structure holding the data for training (`train = true`) or testing (`train = false`)
the neural network. `obs_err_std` is the error standard deviation of the
observations. The variable `lon` is the longitude in degrees east, `lat` is the
latitude in degrees north, `time` is a DateTime vector, `data_full` is a
3-d array with the data and `missingmask` is a boolean mask where true means the data is
missing. `jitter_std` is the standard deviation of the noise to be added to the
data during training.
"""
function NCData(lon,lat,time,data_full,missingmask,ndims;
                train = false,
                obs_err_std = fill(1.,size(data_full,3)),
                jitter_std = fill(0.05,size(data_full,3)),
                ntime_win = 3,
                is3D = false,
                isoutput = (1:size(data_full,3) .== 1),
                cycle_periods = (365.25,), # days
                time_origin = DateTime(1970,1,1),
                remove_mean = true,
                mask = trues(size(data_full)[1:2]),
                direction_obs = nothing,
#                auxdata = (),
                )

    meandata =
        if remove_mean
            sum(x -> (isnan(x) ? zero(x) : x),data_full,dims = 4) ./ sum(.!isnan,data_full,dims = 4)
        else
            zeros(size(data_full)[1:end-1]...,1)
        end

    # number of parameters
    ndata = size(data_full,3)
    ntime = size(data_full,4)

    time_cos = zeros(Float32,length(cycle_periods),length(time))
    time_sin = zeros(Float32,length(cycle_periods),length(time))

    @inbounds for n = 1:length(time)
        # time in days
        t = Dates.value(time[n] - time_origin) / (1000*60*60)

        for k = 1:length(cycle_periods)
            time_cos[k,n] = cos(2π * t/cycle_periods[k])
            time_sin[k,n] = sin(2π * t/cycle_periods[k])
        end
    end

    data = data_full .- meandata
    sz = size(data)

    if direction_obs == nothing
        direction_obs_ = zeros(Float32,sz[1],sz[2],ndata)
    else
        @show size(direction_obs)
        direction_obs_ = direction_obs
    end

    output_ndims = 2

    #if output_ndims == 1
        # dimensions of x: lon, lat, parameter, time, 2
        x = zeros(Float32,(sz[1],sz[2],sz[3],sz[4],2))

        x[:,:,:,:,1] = replace(data,NaN => 0)
        x[:,:,:,:,2] = (1 .- isnan.(data))

        for i = 1:ndata
            # inv. error variance
            x[:,:,i,:,1] ./= obs_err_std[i]^2
            x[:,:,i,:,2] ./= obs_err_std[i]^2
        end
    # else
    #     # dimensions of x: lon, lat, parameter, time, 5
    #     x = zeros(Float32,(sz[1],sz[2],sz[3],sz[4],5))

    #     for i = 1:ndata
    #         # inv. error variance
    #         x[:,:,i,:,1] = sind(direction_obs[:,:,i]) .* replace(data[:,:,i,:],NaN => 0)
    #         x[:,:,i,:,2] = cosd(direction_obs[:,:,i]) .* replace(data[:,:,i,:],NaN => 0)
    #         x[:,:,i,:,3] .= sind(direction_obs[:,:,i]).^2
    #         x[:,:,i,:,4] .= sind(direction_obs[:,:,i]) .* cosd(direction_obs[:,:,i])
    #         x[:,:,i,:,5] .= cosd(direction_obs[:,:,i]).^2
    #     end
    # end
    # scale between -1 and 1
    lon_scaled = Float32.(2 * (lon .- minimum(lon)) / (maximum(lon) - minimum(lon)) .- 1)
    lat_scaled = Float32.(2 * (lat .- minimum(lat)) / (maximum(lat) - minimum(lat)) .- 1)

    N = (is3D ? 4 : 3)

    NCData{Float32,N}(
        Float32.(lon),Float32.(lat),
        time,
        data_full,
        missingmask,
        meandata[:,:,:,1],
        mask,
        x,
        isoutput,
        train,
        Float32.(obs_err_std),
        Float32.(jitter_std),
        lon_scaled,
        lat_scaled,
        time_cos,
        time_sin,
        ntime_win,
        # auxdata,
        direction_obs_,
        output_ndims,
        ndims,
    )
end


getp(x,sym,default) = (hasproperty(x, sym) ? getproperty(x,sym) : default)

function NCData(data; kwargs...)
    lon,lat,datatime,data_full,missingmask,mask = load_gridded_nc(data)

    default_jitter_std = 0.05

    jitter_std = [getp(d,:jitter_std,default_jitter_std) for d in data]
    ndims = [getp(d,:ndims,1) for d in data]

    return NCData(lon,lat,datatime,data_full,missingmask,ndims;
                  obs_err_std = [d.obs_err_std for d in data],
                  jitter_std = jitter_std,
                  isoutput = [d.isoutput for d in data],
                  mask = mask,
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
    ncycles = size(dd.time_cos,1)

    nrange = max.(1, min.(ntime, (-ntime_win_half:ntime_win_half) .+ ind))
    output_ndims = 2

    # metadata
    @inbounds for j = 1:sz[2]
        for i = 1:sz[1]
            xin[i,j,1]  = dd.lon_scaled[i]
            xin[i,j,2]  = dd.lat_scaled[j]

            for k = 1:ncycles
                xin[i,j,2+k]  = dd.time_cos[k,ind]
                xin[i,j,3+k]  = dd.time_sin[k,ind]
            end
        end
    end

    ioffset = 3 + 2*ncycles
    offset = ioffset

#    for aux in dd.auxdata
#        ioffset += 1
#        xin[:,:,ioffset] .= aux
#    end

    #=

    xin[i,j,1:4] = metadata "lon/lat/time (cos+sin)"
    xin[i,j,5:end] order:
      param 1 at time instance 1 (mean/err_var and 1/err_var)
      param 1 at time instance 2 (mean/err_var and 1/err_var)
      param 1 at time instance 3 (mean/err_var and 1/err_var)
      [...]
      param 1 at time instance ntime_win (mean/err_var and 1/err_var)
      param 2 at time instance 1 (mean/err_var and 1/err_var)
      param 2 at time instance 2 (mean/err_var and 1/err_var)
      param 2 at time instance 3 (mean/err_var and 1/err_var)
      [...]
      param 2 at time instance ntime_win (mean/err_var and 1/err_var)
      [...]
      param ndata at time instance ntime_win (mean/err_var and 1/err_var)
    =#

    @inbounds for idata = 1:ndata
        for (localn,n) in enumerate(nrange)
            yield()
            for j = 1:sz[2]
                for i = 1:sz[1]
                    if dd.ndims[idata] == 1
                        xin[i,j,offset  ] = dd.x[i,j,idata,n,1]
                        xin[i,j,offset+1] = dd.x[i,j,idata,n,2]
                    else
                        xin[i,j,offset  ] = sind(dd.direction_obs[i,j,idata]) .* dd.x[i,j,idata,n,1]
                        xin[i,j,offset+1] = cosd(dd.direction_obs[i,j,idata]) .* dd.x[i,j,idata,n,1]
                        xin[i,j,offset+2] = sind(dd.direction_obs[i,j,idata]).^2
                        xin[i,j,offset+3] = sind(dd.direction_obs[i,j,idata]) .* cosd(dd.direction_obs[i,j,idata])
                        xin[i,j,offset+4] = cosd(dd.direction_obs[i,j,idata]).^2
                    end
                end
            end
            offset += nscalar_per_obs_(dd.ndims[idata])
        end
    end

    # add missing data during training randomly to param 1 at the central
    # time step
    if dd.train
        imask = rand(1:size(dd.missingmask,3))
        yield()
        offset = ioffset
        @inbounds for idata = 1:ndata # parameters
            nscalar_per_obs = nscalar_per_obs_(dd.ndims[idata])

            for (localn,n) in enumerate(nrange) # time
                if localn == centraln
                    for k = 0:(nscalar_per_obs-1)
                        for j = 1:sz[2]
                            for i = 1:sz[1]
                                if dd.missingmask[i,j,imask]
                                    xin[i,j,offset+k] = 0
                                end
                            end
                        end
                    end
                end

                for k = 0:(dd.ndims[idata]-1)
                    for j = 1:sz[2]
                        for i = 1:sz[1]
                            # add jitter to mean
                            xin[i,j,offset+k] += (dd.jitter_std[idata] * randn(T))
                        end
                    end
                end
                offset += nscalar_per_obs
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
    ncycles = size(dd.time_cos,1)

    nrange = max.(1, min.(ntime, (-ntime_win_half:ntime_win_half) .+ ind))

    # metadata
    @inbounds for (localn,n) in enumerate(nrange)
        yield()
        for j = 1:sz[2]
            for i = 1:sz[1]
                xin[i,j,localn,1]  = dd.lon_scaled[i]
                xin[i,j,localn,2]  = dd.lat_scaled[j]

                for k = 1:ncycles
                    xin[i,j,localn,2+k]  = dd.time_cos[k,n]
                    xin[i,j,localn,3+k]  = dd.time_sin[k,n]
                end
            end
        end
    end

    # data
    @inbounds for idata = 1:ndata
        yield()
        for (localn,n) in enumerate(nrange)
            for j = 1:sz[2]
                for i = 1:sz[1]
                    offset = 3 + 2*ncycles + (idata-1)*2
                    xin[i,j,localn,offset  ]  = dd.x[i,j,idata,n,1]
                    xin[i,j,localn,offset+1]  = dd.x[i,j,idata,n,2]
                end
            end
        end
    end

    # add missing data during training randomly
    if dd.train
        offset = 3 + 2*ncycles
        for (localn,n) in enumerate(nrange)
            yield()
            imask = rand(1:size(dd.missingmask,3))

            @inbounds for j = 1:sz[2]
                for i = 1:sz[1]
                    if dd.missingmask[i,j,imask]
                        xin[i,j,localn,offset  ] = 0
                        xin[i,j,localn,offset+1] = 0
                    end

                    # add jitter
                    for idata = 1:ndata
                        xin[i,j,localn,offset + (idata-1)*2] += dd.jitter_std[idata] * randn(T)
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

nobs(dd::NCData) = length(dd.time)
function getobs(dd::NCData{T},index::Int) where T
    data = (zeros(T,sizex(dd)),zeros(T,sizey(dd)))
    return getobs!(dd,data,index)
end

function getobs!(dd::NCData,data,index::Int)
    getxy!(dd,index,data[1],data[2])
    return data
end

function savesample(ds,varnames,xrec,meandata,ii,offset;
                    output_ndims = 1,
                    mask = nothing)

    fill_value = -9999.

    function accumulate!(var,index,slice,count)
        # add mask
        var[:,:,index] =
            replace(((count-1) * var[:,:,index] + slice) / count, NaN => fill_value)
    end

    if offset == 0
        ds.attrib["count"] = ds.attrib["count"] + 1
    end
    count = Int(ds.attrib["count"])

    if count == 1
        sz = (size(xrec,1),size(xrec,2))

        for ivar2 = 1:length(varnames)
            nc_batch_m_rec = ds[varnames[ivar2]]
            nc_batch_sigma_rec = ds[varnames[ivar2] * "_error"]

            for n in 1:size(xrec,4)
                nc_batch_m_rec.var[:,:,n+offset] = zeros(Float32,sz)
                nc_batch_sigma_rec.var[:,:,n+offset] = zeros(Float32,sz)
            end

            if output_ndims == 2
                # error covariance
                for ivar1 = (ivar2+1):length(varnames)
                    nc_batch_covar = ds[(varnames[ivar2] * "_" * varnames[ivar1] * "_covar")]
                    for n in 1:size(xrec,4)
                        nc_batch_covar.var[:,:,n+offset] = zeros(Float32,sz)
                    end
                end
            end
        end
    end

    # typically the batch size
    nmax = size(xrec,4)

    if output_ndims == 1
        for (ivar,varname) in enumerate(varnames)
            nc_batch_m_rec = ds[varname]
            nc_batch_sigma_rec = ds[varname * "_error"]

            batch_m_rec = xrec[:,:,2*ivar - 1,:]
            batch_σ2_rec = xrec[:,:,2*ivar,:]

            recdata = batch_m_rec .+ meandata[:,:,ivar]
            batch_sigma_rec = sqrt.(batch_σ2_rec)

            batch_sigma_rec[isnan.(recdata)] .= NaN

            for n in 1:nmax
                if !isnothing(mask)
                    view(recdata,:,:,n)[.!mask] .= NaN
                    view(batch_sigma_rec,:,:,n)[.!mask] .= NaN
                end

                accumulate!(nc_batch_m_rec.var,n+offset,recdata[:,:,n],count)
                accumulate!(nc_batch_sigma_rec.var,n+offset,batch_sigma_rec[:,:,n],count)
            end
        end
    else
        # assume
        @assert all(meandata .== 0)
        @assert output_ndims == 2
        @assert length(varnames) == 2

        P11,P12,P22 = vector2_covariance(xrec)
        Pcov = reshape([P11,P12,P12,P22],2,2)

        #uv = (xrec[:,:,1,:] .* Pcov[1,1],xrec[:,:,2,:] .* Pcov[2,2])
        uv = vector2_mean(xrec,(P11,P12,P22))

        for ivar2 = 1:length(varnames)
            nc_batch_m_rec = ds[varnames[ivar2]]
            nc_batch_sigma_rec = ds[varnames[ivar2] * "_error"]
            for n in 1:nmax
                # accumulate mean
                accumulate!(nc_batch_m_rec.var,n+offset,uv[ivar2][:,:,n],count)

                # error variance
                accumulate!(nc_batch_sigma_rec.var,n+offset,sqrt.(Pcov[ivar2,ivar2][:,:,n]),count)
            end

            # error covariance
            for ivar1 = (ivar2+1):length(varnames)
                nc_batch_covar = ds[(varnames[ivar2] * "_" * varnames[ivar1] * "_covar")]
                for n in 1:nmax
                    accumulate!(nc_batch_covar.var,n+offset,Pcov[ivar1,ivar2][:,:,n],count)
                end
            end
        end
    end
end



function ncsetup(fname,varnames,(lon,lat),meandata; output_ndims = 1)
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


    for (i,varname) in enumerate(varnames)
        nc_batch_m_rec = defVar(
            ds,
            varname, Float32, ("lon","lat","time"),
            fillvalue=fill_value)

        nc_batch_sigma_rec = defVar(
            ds,
            varname * "_error", Float32, ("lon","lat","time"),
            fillvalue=fill_value)

        if output_ndims == 1
            nc_meandata = defVar(
                ds,
                varname * "_mean", Float32, ("lon","lat"),
                fillvalue=fill_value)

            nc_meandata[:,:] = replace(meandata[:,:,i], NaN => missing)
        end
    end

    if output_ndims == 2
        @assert length(varnames) == 2

        defVar(
            ds,
            (varnames[1] * "_" * varnames[2] * "_covar"), Float32, ("lon","lat","time"),
            fillvalue=fill_value)
    end

    # data
    nc_lon[:] = lon
    nc_lat[:] = lat
    ds.attrib["count"] = 0
    return ds
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

#    for ibatch = 1:length(bs)
#    Threads.@threads for ibatch = 1:length(bs)
    ThreadsX.foreach(1:length(bs)) do ibatch
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
