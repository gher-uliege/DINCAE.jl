module AMDGPUExt

import DINCAE: interpnd!, interp_adjn!, _to_device
using AMDGPU
using Flux


function interpnd_d!(pos::AbstractVector{<:NTuple{N}},A,vec) where N
    index = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    stride = gridGroupDim().x * workgroupDim().x

    @inbounds for i = index:stride:length(pos)
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

    return nothing
end

function interpnd!(pos::AbstractVector{<:NTuple{N}},d_A::ROCArray,vec_d) where N
    AMDGPU.@sync begin
        len = length(pos)
        kernel = @roc launch=false interpnd_d!(pos,d_A,vec_d)
        config = launch_configuration(kernel.fun)
        groupsize = min(len, config.groupsize)
        gridsize = cld(len, groupsize)
        @debug gridsize,groupsize

        kernel(pos,d_A,vec_d; groupsize, gridsize)
    end
end


function interp_adjn_d!(pos::AbstractVector{<:NTuple{N}},values,A2) where N
    index = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    stride = gridGroupDim().x * workgroupDim().x

    A2 .= 0

    @inbounds for i = index:stride:length(pos)
        p = pos[i]
        ind = floor.(Int,p)

        # interpolation coefficients
        c = p .- ind

        for offset in CartesianIndices(ntuple(n -> 0:1,Val(N)))
            p2 = Tuple(offset) .+ ind

            cc = prod(ntuple(n -> (offset[n] == 1 ? c[n] : 1-c[n]),Val(N)))

            I = LinearIndices(A2)[p2...]
            AMDGPU.atomic_add!(pointer(A2,I), values[i] * cc)
        end
    end

    return nothing
end


function interp_adjn!(pos::AbstractVector{<:NTuple{N}},values_d::ROCArray,d_A2) where N
    AMDGPU.@sync begin
        len = length(pos)
        #numgridsize = ceil(Int, length(pos)/256)
        # must be one
        numgridsize = 1
        @roc groupsize=256 gridsize=numgridsize interp_adjn_d!(pos,values_d,d_A2)
    end
end

@inline function _to_device(::Type{Atype}) where Atype <: ROCArray
    return Flux.gpu
end

end
