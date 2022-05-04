"""


        ⎛ L11   0  ⎞
    L = ⎝ L21  L22 ⎠

    P = L * L'

        ⎛ P11  P12 ⎞
    P = ⎝ P12  P22 ⎠

"""
function vector2_covariance(xrec; obsoper = identity)
    L11 = obsoper(xrec[:,:,3,:])
    L21 = obsoper(xrec[:,:,4,:])
    L22 = obsoper(xrec[:,:,5,:])


    P11 = L11.^2
    P12 = L11 .* L21
    P22 = L21.^2 + L22.^2

    return (P11,P12,P22)
end

function vector2_costfun(xrec,xtrue,truth_uncertain,directionobs)
    nsites = size(directionobs,3)

    # interpolate to location of observations
    # output should be the same shape as m_true
    obsoper(x::AbstractArray{T,2}) where T = repeat(x,(1,1,nsites))
    #obsoper(x::AbstractArray{T,3}) where T = mapslices(obsoper,x,dims=(1,2))
    #obsoper(x::AbstractArray{T,3}) where T = x
    function obsoper(x::AbstractArray{T,3}) where T
        tmp = reshape(x,(size(x)[1:2]...,1,size(x,3)))
        repeat(tmp,inner=(1,1,nsites,1))
    end

    N = ndims(xtrue)
    allst = ntuple(i -> :, N-2)

    # u/v at observations location
    uv_obs = (obsoper(xrec[:,:,1,:]), obsoper(xrec[:,:,2,:]))

    projection_vector = (sind.(directionobs), cosd.(directionobs))

    # projected radial currents
    m_rec = (projection_vector[1] .* uv_obs[1]
             + projection_vector[2] .* uv_obs[2])


    P11,P12,P22 = vector2_covariance(xrec; obsoper = obsoper)

    σ2_rec = @. (projection_vector[1]^2 * P11
                 + 2 * projection_vector[1]*projection_vector[2] * P12
                 + projection_vector[2]^2 * P22)

    @debug begin
        nneg = count(σ2_rec .< 0)
        "number of negative value $nneg"
    end

    σ2_true = sinv(xtrue[allst...,2:2:2*nsites,:])
    m_true = xtrue[allst...,1:2:2*nsites,:] .* σ2_true

    # # 1 if measurement
    # # 0 if no measurement (cloud or land for SST)
    mask_noncloud = xtrue[allst...,2:2:2*nsites,:] .!= 0

    cost = costfun_single(m_rec,σ2_rec,m_true,σ2_true,mask_noncloud,truth_uncertain)
    return cost
end



function (model::ModelVector2_1)(x_)
    N = ndims(x_)
    allst = ntuple(i -> :, N-2)

    x = model.chain(x_)
    return x
end



function (model::ModelVector2_1)(inputs_,xtrue)
    xrec = model(inputs_)
    return vector2_costfun(xrec,xtrue,model.truth_uncertain,model.directionobs)
end
