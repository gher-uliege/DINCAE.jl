module CUDAExt

import DINCAE: interpnd!, interp_adjn!, _to_device
using CUDA
using Flux


function interpnd_d!(pos::AbstractVector{<:NTuple{N}},A,vec) where N
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

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

function interpnd!(pos::AbstractVector{<:NTuple{N}},d_A::CuArray,vec_d) where N
    CUDA.@sync begin
        len = length(pos)
        kernel = @cuda launch=false interpnd_d!(pos,d_A,vec_d)
        config = launch_configuration(kernel.fun)
        threads = min(len, config.threads)
        blocks = cld(len, threads)
        @debug blocks,threads

        kernel(pos,d_A,vec_d; threads, blocks)
    end
end


function interp_adjn_d!(pos::AbstractVector{<:NTuple{N}},values,A2) where N
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    @inbounds for i = index:stride:length(pos)
        p = pos[i]
        ind = floor.(Int,p)

        # interpolation coefficients
        c = p .- ind

        for offset in CartesianIndices(ntuple(n -> 0:1,Val(N)))
            p2 = Tuple(offset) .+ ind

            cc = prod(ntuple(n -> (offset[n] == 1 ? c[n] : 1-c[n]),Val(N)))

            I = LinearIndices(A2)[p2...]
            CUDA.atomic_add!(pointer(A2,I), values[i] * cc)
        end
    end

    return nothing
end


function interp_adjn!(pos::AbstractVector{<:NTuple{N}},cuvalues::CuArray,d_A2) where N
    d_A2 .= 0

    CUDA.@sync begin
        len = length(pos)
        #numblocks = ceil(Int, length(pos)/256)
        # must be one
        numblocks = 1
        @cuda threads=256 blocks=numblocks interp_adjn_d!(pos,cuvalues,d_A2)
    end
end

@inline function _to_device(::Type{Atype}) where Atype <: CuArray
    return Flux.gpu
end

end
