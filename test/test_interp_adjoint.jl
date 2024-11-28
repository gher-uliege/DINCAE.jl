using AMDGPU
using CUDA
using DINCAE: interpnd, interp_adjn
using Flux
using LinearAlgebra
using Test

f(pos,B) = sum(interpnd(pos,B))

device_list = Any[identity] # CPU

if CUDA.functional()
    push!(device_list,cu)
end

if AMDGPU.functional()
    push!(device_list,roc)
end

pos = [(1.5f0,2.f0),(5.f0,2.3f0)]
values = [3.f0,4.f0]
sz = (6,7)
B = ones(Float32,sz)

A_ref = zeros(Float32,sz)
A_ref[1,2] = A_ref[2,2] = 0.5 * 3
A_ref[5,2] = 0.7 * 4
A_ref[5,3] = 0.3 * 4

dA_ref = zeros(Float32,sz)
dA_ref[1,2] = dA_ref[2,2] = 0.5
dA_ref[5,2] = 0.7
dA_ref[5,3] = 0.3


for device in device_list
    pos_d  = device(pos)
    values_d = device(values)
    B_d = device(B)

    A_d = interp_adjn(pos_d,values_d,sz)
    @test Array(A_d) ≈ A_ref

    dA_d = gradient(f,pos_d,B_d)[2]

    @test Array(dA_d) ≈ dA_ref
end

# check adjoint relationship

A = randn(Float32,sz)
dAi = randn(Float32,length(pos))

@test dAi ⋅ interpnd(pos,A) ≈ A ⋅ interp_adjn(pos,dAi,size(A))

for device in device_list[2:end]
    A_d = device(A)
    dAi_d = device(dAi)
    pos_d = device(pos)

    @test interpnd(pos,A) ≈ Array(interpnd(pos_d,A_d))
    @test interp_adjn(pos,dAi,size(A)) ≈ Array(interp_adjn(pos_d,dAi_d,size(A)))

    @test dAi_d ⋅ interpnd(pos_d,A_d) ≈ A_d ⋅ interp_adjn(pos_d,dAi_d,size(A))
end
