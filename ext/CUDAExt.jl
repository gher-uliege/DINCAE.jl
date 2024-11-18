module CUDAExt

import DINCAE: interpnd!, interp_adjn!, _to_device
using CUDA
using Flux


function cu_interpnd!(pos::AbstractVector{<:NTuple{N}},A,vec) where N
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

function interpnd!(pos::AbstractVector{<:NTuple{N}},cuA::CuArray,cuvec) where N
    CUDA.@sync begin
        len = length(pos)
        kernel = @cuda launch=false cu_interpnd!(pos,cuA,cuvec)
        config = launch_configuration(kernel.fun)
        threads = min(len, config.threads)
        blocks = cld(len, threads)

        kernel(pos,cuA,cuvec; threads, blocks)
    end
end


function cu_interp_adjn!(pos::AbstractVector{<:NTuple{N}},values,A2) where N
#    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#    stride = gridDim().x * blockDim().x
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x

    A2 .= 0

    @inbounds for i = index:stride:length(pos)
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

    return nothing
end


function interp_adjn!(pos::AbstractVector{<:NTuple{N}},cuvalues::CuArray,cuA2) where N

    CUDA.@sync begin
        @cuda threads=256 cu_interp_adjn!(pos,cuvalues,cuA2)

        #=
        len = length(pos)
        kernel = @cuda launch=false cu_interp_adjn!(pos,cuvalues,cuA2)
        config = launch_configuration(kernel.fun)
        threads = min(len, config.threads)
        blocks = cld(len, threads)

        kernel(pos,cuvalues,cuA2; threads, blocks)
=#
    end

end

@inline function _to_device(::Type{Atype}) where Atype <: CuArray
    return Flux.gpu
end

end
