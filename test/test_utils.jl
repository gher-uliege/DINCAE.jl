using DINCAE
using Test

a = [1,2,3]
@test DINCAE.normalize2(a) == [-1,0,1]

a = zeros(10,10)
@test all(isfinite,DINCAE.normalize2(a))


a = ones(2,3)
@test DINCAE.SumSkip(x -> 2*x)(a) == 3*a
@test DINCAE.CatSkip(x -> 2*x)(a) == cat(2*a,a,dims=3)
