module AMDGPUExt

import DINCAE: interpnd!, interp_adjn!, _to_device
using AMDGPU
using Flux


function interpnd_d!(pos::AbstractVector{<:NTuple{N}},A,vec) where N
    index = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    stride = gridGroupDim().x * workgroupDim().x

    @inbounds for i = index:stride:length(pos)
        p = pos[i]
        #ind = floor.(Int,p)
        ind = unsafe_trunc.(Int32,floor.(p))

        # interpolation coefficients
        #c = p .- ind
        c = ntuple(Val(N)) do i
            p[i] - ind[i]
        end

        for offset in CartesianIndices(ntuple(n -> 0:1,Val(N)))
            #p2 = Tuple(offset) .+ ind
            p2 = ntuple(Val(N)) do i
                offset[i] + ind[i]
            end

            cc = prod(ntuple(n -> (offset[n] == 1 ? c[n] : 1-c[n]),Val(N)))
            vec[i] += A[p2...] * cc
        end
    end

    return nothing
end

function interpnd!(pos::AbstractVector{<:NTuple{N}},A::ROCArray,vec) where N
    AMDGPU.@sync begin
        len = length(pos)
        kernel = @roc launch=false interpnd_d!(pos,A,vec)
        config = AMDGPU.launch_configuration(kernel)
        groupsize = min(len, config.groupsize)
        gridsize = cld(len, groupsize)
        @debug gridsize,groupsize

        kernel(pos,A,vec; groupsize, gridsize)
    end
end


function interp_adjn_d!(pos::AbstractVector{<:NTuple{N}},values,B) where N
    index = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    stride = gridGroupDim().x * workgroupDim().x

    @inbounds for i = index:stride:length(pos)
        p = pos[i]
        #ind = floor.(Int,p)
        ind = unsafe_trunc.(Int32,floor.(p))

        # interpolation coefficients
        #c = p .- ind
        c = ntuple(Val(N)) do i
            p[i] - ind[i]
        end

        for offset in CartesianIndices(ntuple(n -> 0:1,Val(N)))
            # p2 = Tuple(offset) .+ ind
            p2 = ntuple(Val(N)) do i
                offset[i] + ind[i]
            end

            cc = prod(ntuple(n -> (offset[n] == 1 ? c[n] : 1-c[n]),Val(N)))

            I = LinearIndices(B)[p2...]
            B[I] += values[i] * cc
        end
    end

    return nothing
end


function interp_adjn!(pos::AbstractVector{<:NTuple{N}},values::ROCArray,B) where N
    B .= 0

    AMDGPU.@sync begin
        len = length(pos)
        #numgridsize = ceil(Int, length(pos)/256)
        # must be one
        numgridsize = 1
        @roc groupsize=256 gridsize=numgridsize interp_adjn_d!(pos,values,B)
    end
end

@inline function _to_device(::Type{Atype}) where Atype <: ROCArray
    return Flux.gpu
end

end
