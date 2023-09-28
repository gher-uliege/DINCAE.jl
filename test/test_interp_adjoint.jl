using CUDA
using DINCAE: interpnd, interp_adjn
using Flux
using Test
using LinearAlgebra

f(pos,B) = sum(interpnd(pos,B))

pos = [(1.5f0,2.f0),(5.f0,2.3f0)]
values = [3.f0,4.f0]
sz = (6,7)
B = ones(Float32,sz)

A = interp_adjn(pos,values,sz)

@test A[1,2] ≈ 1.5
@test A[2,2] ≈ 1.5

dA = gradient(f,pos,B)[2]

@test dA[1,2] ≈ 0.5
@test dA[2,2] ≈ 0.5

if CUDA.functional()
    cu_pos  = cu(pos)
    cu_values = cu(values)
    cu_B = cu(B)

    cu_A = interp_adjn(cu_pos,cu_values,sz)

    CUDA.@allowscalar begin
        @test cu_A[1,2] ≈ 1.5
        @test cu_A[2,2] ≈ 1.5
    end

    cu_dA = gradient(f,cu_pos,cu_B)[2]


    CUDA.@allowscalar begin
        @test cu_dA[1,2] ≈ 0.5
        @test cu_dA[2,2] ≈ 0.5
    end
end

# check adjoint relationship

A = randn(Float32,size(A,1),size(A,2))
Ai = randn(Float32,length(pos))

@test Ai ⋅ interpnd(pos,A) ≈ A ⋅ interp_adjn(pos,Ai,size(A))
