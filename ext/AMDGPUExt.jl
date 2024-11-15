module AMDGPUExt

import DINCAE: interpnd!, interp_adjn!, _to_device
using AMDGPU
using Flux


function interpnd!(pos::AbstractVector{<:NTuple{N}},A::ROCArray,vec) where N
    @roc interpnd!(pos,A,vec)
end

function interp_adjn!(pos::AbstractVector{<:NTuple{N}},values::ROCArray,A2) where N
    @roc interp_adjn!(pos,values,A2)
end

@inline function _to_device(::Type{Atype}) where Atype <: ROCArray
    return Flux.gpu
end

end
