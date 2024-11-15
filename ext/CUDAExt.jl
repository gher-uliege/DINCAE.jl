module CUDAExt

import DINCAE: interpnd!, interp_adjn!, _to_device
using CUDA
using Flux


function interpnd!(pos::AbstractVector{<:NTuple{N}},cuA::CuArray,cuvec) where N
    @cuda interpnd!(pos,cuA,cuvec)
end

function interp_adjn!(pos::AbstractVector{<:NTuple{N}},cuvalues::CuArray,cuA2) where N
    @cuda interp_adjn!(pos,cuvalues,cuA2)
end

@inline function _to_device(::Type{Atype}) where Atype <: CuArray
    return Flux.gpu
end

end
