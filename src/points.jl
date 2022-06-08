"""
     interpnd! and interp_adj! are adjoints

    vec must be zero initially
    """
function interpnd!(pos::AbstractVector{<:NTuple{N}},A,vec) where N
    #@show typeof(A),typeof(vec)

    @inbounds for i = 1:length(pos)
        p = pos[i]
        ind = floor.(Int,p)

        # interpolation coefficients
        c = p .- ind

        for offset in CartesianIndices(ntuple(n -> 0:1,Val(N)))
            p2 = Tuple(offset) .+ ind

            cc = prod(ntuple(n -> (offset[n] == 1 ? c[n] : 1-c[n]),Val(N)))
            vec[i] += A[p2...] * cc
        end
    end

    #return vec
    return nothing
end

function interpnd!(pos::AbstractVector{<:NTuple{N}},A::KnetArray,vec) where N
    cuA = CuArray(A)
    cuvec = CuArray(vec)
    @cuda DINCAE.interpnd!(pos,cuA,cuvec)
end

function interpnd!(pos::AbstractVector{<:NTuple{N}},cuA::CuArray,cuvec) where N
    @cuda DINCAE.interpnd!(pos,cuA,cuvec)
end

#interpnd(pos,A) = interpnd!(pos,A,zeros(eltype(A),length(pos)))
function interpnd(pos,A)
    vec = 0 * A[1:length(pos)]
    #vec = similar(A,length(pos))
    #vec .= 0
    interpnd!(pos,A,vec)
    return vec
end



Knet.AutoGrad.@primitive interpnd(pos,A),dy,y 0 interp_adjn(pos,dy,size(A))

"""
        all positions should be within the domain. exclusive upper bound

    for all i and n
    (1 <= pos[i][n] < sz[n])

    """
function interp_adjn!(pos::AbstractVector{<:NTuple{N}},values,A2) where N
    A2 .= 0

    @inbounds for i = 1:length(values)
        p = pos[i]
        ind = floor.(Int,p)

        # interpolation coefficients
        c = p .- ind

        for offset in CartesianIndices(ntuple(n -> 0:1,Val(N)))
            p2 = Tuple(offset) .+ ind

            cc = prod(ntuple(n -> (offset[n] == 1 ? c[n] : 1-c[n]),Val(N)))
            A2[p2...] += values[i] * cc
        end
    end

    #return A2
    return nothing
end


function interp_adjn!(pos::AbstractVector{<:NTuple{N}},values::KnetArray,A2) where N
    cuvalues = CuArray(values)
    cuA2 = CuArray(A2)
    @cuda interp_adjn!(pos,cuvalues,cuA2)
end

function interp_adjn!(pos::AbstractVector{<:NTuple{N}},cuvalues::CuArray,cuA2) where N
    @cuda interp_adjn!(pos,cuvalues,cuA2)
end

function interp_adjn(pos::AbstractVector{<:NTuple{N}},values,sz::NTuple{N,Int}) where N
    A2 = similar(values,sz)
    interp_adjn!(pos,values,A2)
    return A2
end


function filter_domain!(pos,sz,sel)
    for i = 1:length(values)
        p = pos[i]
        ind = floor.(Int,p)
        sel[i] = true

        for n = 1:length(sz)
            # exclusive upper bound is important, because one is added to indices
            sel[i] = sel[i] & (1 <= ind[n] < sz[n])
        end
    end
    return sel
end

filter_domain(pos,sz) = filter_domain!(pos,sz,falses(length(pos)))



function posindices!(::Type{T},grid::NTuple{N},coord,pos,attribs) where {T,N}
    sz = length.(grid)
    ntime = length(coord[1])
    inside = Vector{Vector{Bool}}(undef,ntime)

    for iday = 1:ntime

        npoints = length(coord[1][iday])


        pos[iday] = Vector{NTuple{N,T}}(undef,npoints)
        inside[iday] = Vector{Bool}(undef,npoints)

        for ipoint = 1:npoints
            pos_ = ntuple(l -> length(grid[l]) * (coord[l][iday][ipoint]-grid[l][1])/(grid[l][end]-grid[l][1]) + 1, Val(N))
            inside_ = true

            for icoord = 1:N
                # exclusive upper bound is important, because one is added to indices
                inside_ = inside_ & (1 <= pos_[icoord] < sz[icoord])
            end

            pos[iday][ipoint] = pos_
            inside[iday][ipoint] = inside_

        end

        # select all data within the domain
        pos[iday] = pos[iday][inside[iday]]
        for attrib in attribs
            attrib[iday] = attrib[iday][inside[iday]]
        end
        for p in coord
            p[iday] = p[iday][inside[iday]]
        end
    end
end


struct PointCloud{Atype,T,N}
    grid_scaled::NTuple{N,Vector{T}}
    dayofyear_cos::Vector{T}
    dayofyear_sin::Vector{T}
    ntime_win::Int
    probability_skip_for_training::T
    train::Bool
    pos::Vector{Vector{NTuple{N,T}}}
    jitter_std_pos::NTuple{N,T}
    x::Vector{Array{T,2}}
    id::Vector{Vector{Int}}
    batch_size::Int
    perm::Vector{Int}
    auxdata::Array{T,4} # lon,lat,nparam,time
end

function PointCloud(
    T,Atype,values,coord,dtime,id,dates,grid;
    train = true,
    ntime_win = 3,
    probability_skip_for_training = 0.2,
    jitter_std_pos = (5f0,5f0),
    auxdata = zeros(T,length(grid[1]),length(grid[2]),0,length(dates)),
    batch_size = 64)

    sz = length.(grid)
    N = length(coord)

    j = 1

    l = 1

    ntime = length(coord[1])
    pos = Vector{Vector{NTuple{N,T}}}(undef,ntime)


    @info "number of provided data points: $(sum(length.(values)))"

    posindices!(T,grid,coord,pos,(values,dtime,id))

    @info "number of data points within domain: $(sum(length.(values)))"

    ndata = 1

    obs_err_std = [1.]
    idata = 1

    x = Vector{Array{T,2}}(undef,ntime)
    for iday = 1:ntime
        x[iday] = zeros(T,length(values[iday]),2)

        x[iday][:,1] = replace(values[iday], NaN => 0) / obs_err_std[idata]^2
        x[iday][:,2] = 1 .- isnan.(values[iday]) / obs_err_std[idata]^2
    end

    ind = 1


    grid_scaled = ntuple(l ->
                         T.(2 * (grid[l] .- minimum(grid[l])) / (maximum(grid[l]) - minimum(grid[l])) .- 1),Val(N))


    year_length = 365.25 # days
    dayofyear_ = Dates.dayofyear.(dates)
    dayofyear_cos = T.(cos.(2π * dayofyear_/year_length))
    dayofyear_sin = T.(sin.(2π * dayofyear_/year_length))

    perm =
        if train
            randperm(ntime)
        else
            1:ntime
        end

    train = PointCloud{Atype,T,2}(
        grid_scaled,
        dayofyear_cos,
        dayofyear_sin,
        ntime_win,
        T(probability_skip_for_training),
        train,
        pos,
        jitter_std_pos,
        x,
        id,
        batch_size,
        perm,
        auxdata,
    )
    return train
end

function pertpos(pos_::Vector{NTuple{N,T}},jitter_std_pos,maxpos::NTuple{N,Int}) where {N,T}
    pp = similar(pos_)
    @inbounds for (i,p) in enumerate(pos_)
        pp[i] = ntuple(l -> clamp(p[l] + jitter_std_pos[l] * randn(T),1,maxpos[l]),Val(N))
    end
    return pp
end

function getxy!(d::PointCloud{Atype,T,N},ind,xin,xtrue) where {Atype,T,N}
    ntime = length(d.x)
    sz = size(xin)

    lon_scaled,lat_scaled = d.grid_scaled
    @inbounds for j = 1:sz[2]
        for i = 1:sz[1]
            #xin[i,j,1] = d.grid_scaled[1][i]
            #xin[i,j,2] = d.grid_scaled[2][j]
            xin[i,j,1] = lon_scaled[i]
            xin[i,j,2] = lat_scaled[j]
            xin[i,j,3] = d.dayofyear_cos[ind]
            xin[i,j,4] = d.dayofyear_sin[ind]
        end
    end

    @inbounds for l = 1:size(d.auxdata,3)
        for j = 1:sz[2]
            for i = 1:sz[1]
                xin[i,j,4+l  ] = d.auxdata[i,j,l,ind]
            end
        end
    end

    idata = 1

    ntime_win = d.ntime_win
    pos = d.pos
    x = d.x

    centraln = (ntime_win+1) ÷ 2
    ntime_win_half = (ntime_win-1) ÷ 2
    nrange = max.(1, min.(ntime, (-ntime_win_half:ntime_win_half) .+ ind))

    unique_id = unique(d.id[ind])


    for (localn,n) in enumerate(nrange)
        offset = 5 + size(d.auxdata,3) + 2*(localn-1) + (idata-1)*2*ntime_win

        pos_ = copy(pos[n])
        npoints = length(pos_)

        if d.train
            pos_ = pertpos(pos_,d.jitter_std_pos,ntuple(l -> sz[l]-1, Val(N)))
        end

        if localn .== centraln
            #  remove data during training randomly

            sel = trues(npoints)
            if d.train

                for uid in unique_id
                    if rand() < d.probability_skip_for_training
                        sel[d.id[ind] .== uid] .= false
                    end
                end

                # at least one track should be there
                if sum(sel) == 0
                    sel[d.id[ind] .== rand(unique_id)] .= true
                end
            end

            interp_adjn!(pos_[sel],x[n][:,1][sel],view(xin,:,:,offset));
            interp_adjn!(pos_[sel],x[n][:,2][sel],view(xin,:,:,offset+1));
        else
            interp_adjn!(pos_,x[n][:,1],view(xin,:,:,offset));
            interp_adjn!(pos_,x[n][:,2],view(xin,:,:,offset+1));
        end
    end

    xtrue[1] = (pos = pos[ind],x = x[ind])

    return (xin,xtrue)
end


function _to_device(::Type{Atype},pos) where Atype <: Union{KnetArray,CuArray}
    return cu(pos)
end

function _to_device(::Type{Atype},pos) where Atype
    return pos
end


function Base.iterate(d::PointCloud{Atype,T,N},index = 0) where {Atype,T,N}

    sz = sizex(d)
    ntime = length(d.x)
    bs = index+1 : min(index + d.batch_size,ntime)

    if length(bs) == 0
        return nothing
    end

    xin = zeros(T,sz...,length(bs))
    xtrue = Vector{NamedTuple{(:pos, :x),Tuple{Vector{Tuple{T,T}},Array{T,N}}}}(undef,length(bs))

    Threads.@threads for ibatch = 1:length(bs)
        j = @inbounds bs[ibatch]

        @inbounds getxy!(d,d.perm[j],
               (@view xin[:,:,:,ibatch]),
               (@view xtrue[ibatch:ibatch]))
    end

    #@show index
    return ((Atype(xin),
             #map(xt -> (pos = xt.pos, x = Atype(xt.x)),xtrue)),bs[end])
             map(xt -> (pos = _to_device(Atype,xt.pos), x = Atype(xt.x)),xtrue)),bs[end])
             #map(xt -> (pos = cu(xt.pos), x = Atype(xt.x)),xtrue)),bs[end])

end

function DINCAE.sizex(d::PointCloud)
    sz = length.(d.grid_scaled)
    ndata = 1
    ntime_win = d.ntime_win

    nvar = 4 + ndata*2*d.ntime_win + size(d.auxdata)[end-1]
    return (sz[1],sz[2],nvar)
end

function costfun(xrec,xtrue::Vector{NamedTuple{(:pos, :x),Tuple{Tpos,TA}}},truth_uncertain) where TA <: Union{Array{T,N},KnetArray{T,N}} where Tpos <: AbstractVector{NTuple{N,T}} where {N,T}

    #@show typeof(xin)
    #@show typeof(xrec)

    batch_size = size(xrec)[end]

    # interpolate to observation location
    #xrec

    ibatch = 1

    #xrec_interp = Vector{Atype{2}}(undef,batch_size)
    xrec_interp = Vector{Any}(undef,batch_size)

    for ibatch = 1:batch_size

        #xrec_interp_ = Atype{2}(undef,length(xtrue[ibatch].pos),2)
        #xrec_interp_ = Array{Any,2}(undef,length(xtrue[ibatch].pos),2)

        # if typeof(xrec) <: AutoGrad.Result
        #     xrec_interp_ = AutoGrad.Result{TA}(undef,length(xtrue[ibatch].pos),2)
        # else
        #     xrec_interp_ = similar(xrec,length(xtrue[ibatch].pos),2)
        # end

        # xrec_interp_ .= 0
        # @show typeof(xrec_interp_)

        # # function interpnd!(pos::AbstractVector{<:NTuple{N}},A,vec) where N

        # interpnd!(xtrue[ibatch].pos,xrec[:,:,1,ibatch],view(xrec_interp_,:,1))
        # interpnd!(xtrue[ibatch].pos,xrec[:,:,2,ibatch],view(xrec_interp_,:,2))

        # xrec_interp[ibatch] = xrec_interp_

        xrec_interp[ibatch] =
            hcat(interpnd(xtrue[ibatch].pos,xrec[:,:,1,ibatch]),
                 interpnd(xtrue[ibatch].pos,xrec[:,:,2,ibatch]))
    end

    xrec_interp2 = reduce(vcat,xrec_interp)

    #@show size(xrec_interp2)
    #@show typeof(xrec_interp2)

    #@show map(typeof,xtrue)
    xtrue_interp2 = reduce(vcat,map(xt -> xt.x,xtrue))
    cost = DINCAE.costfun(xrec_interp2[:,:,1:1],xtrue_interp2[:,:,1:1],truth_uncertain)

    return cost
end


function loaddata(filename,varname)
    Dataset(filename) do ds
        values = loadragged(ds[varname],:);
        lon = loadragged(ds["lon"],:);
        lat = loadragged(ds["lat"],:);
        dtime = loadragged(ds["dtime"],:);
        id = loadragged(ds["id"],:);

        values = copy.(values)
        lon = copy.(lon)
        lat = copy.(lat)
        id = copy.(id)
        dtime = copy.(dtime)

        dates = nomissing(ds["dates"][:]);
        return values,(lon,lat),dtime,id,dates
    end
end


"""
    DINCAE.reconstruct_points(T,Atype,filename,varname,grid,fnames_rec )



Mandatory parameters:

* `T`: `Float32` or `Float64`: float-type used by the neural network
* `Array{T}` or `KnetArray{T}`: array-type used by the neural network.
* `filename`: NetCDF file in the format described below.
* `varname`: name of the primary variable in the NetCDF file
* `grid`: tuple of ranges with the grid in the longitude and latitude direction e.g. `(-180:1:180,-90:1:90)`.
* `fnames_rec`: NetCDF file names of the reconstruction

Optional parameters:
* `jitter_std_pos`: standard deviation of the noise to be added to the position of observation (default `(5,5)`)
* `auxdata_files`: gridded auxillary data file for a multivariate reconstruction. `auxdata_files` is an array of named tuples with the fields (`filename`, the file name of the NetCDF file, `varname` the NetCDF name of the primary variable and `errvarname` the NetCDF name of the expected standard deviation error). For example:

```
auxdata_files = [
  (filename = "big-sst-file.nc"),
   varname = "SST",
   errvarname = "SST_error")]
```

The data in the file should already be interpolated on the targed grid. The file structure of the NetCDF file is described in `DINCAE.load_gridded_nc`.

See `DINCAE.reconstruct` for other optional parameters.

An (minimal) example of the NetCDF file is:

```
netcdf all-sla.train {
dimensions:
	track = 9628 ;
	time = 7445528 ;
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
```

The file should contain the variables `lon` (longitude), `lat` (latitude), `dtime` (time of measurement) and `id` (numeric identifier, only used by post processing scripts) and `dates` (time instance of the gridded field). The file should be in the [contiguous ragged array representation](https://cfconventions.org/cf-conventions/cf-conventions.html#_contiguous_ragged_array_representation) as specified by the CF convention allowing to group data points into "features" (e.g. tracks for altimetry). Every feature can also contain a single data point.



"""
function reconstruct_points(
    T,Atype,filename,varname,grid,fnames_rec;
    epochs = 60,
    batch_size = 32,
    truth_uncertain = false,
    enc_nfilter_internal = [10,20,30,40,50],
    skipconnections = 2:(length(enc_nfilter_internal)+1),
    clip_grad = 5.0,
    regularization_L1_beta = 0,
    regularization_L2_beta = 0,
    save_epochs = 60:10:epochs,
    upsampling_method = :nearest,
    probability_skip_for_training = 0.2,
    jitter_std_pos = (5f0,5f0),
    ntime_win = 9,
    learning_rate = 0.001,
    learning_rate_decay_epoch = Inf,
    min_std_err = 0.006737946999085467,
    loss_weights_refine = (1.,),
    auxdata_files = [],
    savesnapshot = false,
)

    fname_rec = fnames_rec[1]

    values,coord,dtime,id,dates = DINCAE.loaddata(filename,varname)

    #values,coord,dtime,id,dates = values[1:100],(coord[1][1:100],coord[2][1:100]),dtime[1:100],id[1:100],dates[1:100]

    auxdata = load_aux_data(T,(length.(grid)...,length(dates)),auxdata_files)

    train = DINCAE.PointCloud(
        T,Atype,values,coord,dtime,id,dates,grid;
        batch_size = batch_size,
        ntime_win = ntime_win,
        jitter_std_pos = jitter_std_pos,
        probability_skip_for_training = probability_skip_for_training,
        auxdata = auxdata,
        train = true)

    data_iter = [
        train,
        DINCAE.PointCloud(
            T,Atype,values,coord,dtime,id,dates,grid;
            batch_size = batch_size,
            ntime_win = ntime_win,
            jitter_std_pos = jitter_std_pos,
            probability_skip_for_training = probability_skip_for_training,
            auxdata = auxdata,
            train = false)
    ]

    sz = DINCAE.sizex(train)

    nvar = sz[3]
    @info "number of variaiables: $nvar"
    gamma = log(min_std_err^(-2))
    @info "gamma: $gamma"
    noutput = 1

    enc_nfilter = vcat([nvar],enc_nfilter_internal)
    dec_nfilter = vcat([2*noutput],enc_nfilter_internal)

    if loss_weights_refine == (1.,)
        println("no refine")
        steps = (DINCAE.recmodel4(sz[1:2],enc_nfilter,dec_nfilter,skipconnections; method = upsampling_method) ,)
    else
        enc_nfilter2 = copy(enc_nfilter)
        enc_nfilter2[1] += dec_nfilter[1]
        dec_nfilter2 = copy(enc_nfilter2)

        steps = (DINCAE.recmodel4(sz[1:2],enc_nfilter,dec_nfilter,skipconnections; method = upsampling_method),
                 DINCAE.recmodel4(sz[1:2],enc_nfilter2,dec_nfilter2,skipconnections; method = upsampling_method))
    end
    model = StepModel(
        steps,loss_weights_refine,truth_uncertain,gamma;
        regularization_L1 = regularization_L1_beta,
        regularization_L2 = regularization_L2_beta,
    )

    #=
    i = 5
    scatter(lon[i],lat[i],1,sla[i])
    =#


    xin,xtrue = first(train)

    # test forward pass
    @show sum(model(xin))
    @show size(model(xin))
    @show sz
    # test loss function
    @show model(xin,xtrue)

    losses = []


    if isfile(fname_rec)
        @info "delete $fname_rec"
    end

    meandata = zeros(sz[1:2]);
    ds = ncsetup(fname_rec,[varname],(grid[1],grid[2]),meandata)

    @time for e = 1:epochs
        #@time @profile for e = 1:epochs
        #Juno.@profile @time  for e = 1:epochs

        lr = learning_rate * 0.5^(e / learning_rate_decay_epoch)

        # loop over training datasets
        for (ii,loss) in enumerate(adam(model, train; gclip = clip_grad, lr = lr))
            push!(losses,loss)

            if (ii-1) % 20 == 0
                println("epoch: $(@sprintf("%5d",e )) loss $(@sprintf("%5.4f",loss)) $lr")
                flush(stdout)
            end
        end

        if e ∈ save_epochs
            println("Save output $e")

            if savesnapshot
                fname_rec_snapshot = replace(fname_rec,".nc" => "-epoch$(@sprintf("%05d",e )).nc")
                ds_snapshot = ncsetup(fname_rec_snapshot,[varname],(grid[1],grid[2]),meandata)
            end

            @time for (ii,(inputs_,xtrue)) in enumerate(data_iter[2])
                xrec = Array(model(inputs_))
                offset = (ii-1)*batch_size
                savesample(ds,[varname],xrec,meandata,ii-1,offset)

                if savesnapshot
                    savesample(ds_snapshot,[varname],xrec,meandata,ii-1,offset)
                end
            end

            if savesnapshot
                close(ds_snapshot)
            end

            sync(ds)
        end
    end
    close(ds)
end
