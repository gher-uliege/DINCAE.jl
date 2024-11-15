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
    pos_d  = cu(pos)
    values_d = cu(values)
    B_d = cu(B)

    A_d = interp_adjn(pos_d,values_d,sz)

    CUDA.@allowscalar begin
        @test A_d[1,2] ≈ 1.5
        @test A_d[2,2] ≈ 1.5
    end

    dA_d = gradient(f,pos_d,B_d)[2]


    CUDA.@allowscalar begin
        @test dA_d[1,2] ≈ 0.5
        @test dA_d[2,2] ≈ 0.5
    end
end

# check adjoint relationship

A = randn(Float32,size(A,1),size(A,2))
Ai = randn(Float32,length(pos))

@test Ai ⋅ interpnd(pos,A) ≈ A ⋅ interp_adjn(pos,Ai,size(A))
