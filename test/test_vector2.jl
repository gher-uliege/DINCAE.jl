# TODO:
# add sin/cos of direction to xin
# maybe as auxiliary variables?

using Test
using DINCAE
using CUDA

sz = (100,120)
batch_size = 3

xrec = randn(sz...,5,batch_size)

nsites = 3
xtrue = randn(sz...,2*nsites,batch_size)

# direction in degress from 0 to 360
directionobs = 360 * rand(sz...,nsites,batch_size)

truth_uncertain = false


cost = @time DINCAE.vector2_costfun(xrec,xtrue,truth_uncertain,directionobs)

@test cost isa Number


if CUDA.functional()
    # test on GPU
    for TA = [CuArray]
        cost_TA = DINCAE.vector2_costfun(TA(xrec),TA(xtrue),truth_uncertain,TA(directionobs))
        @test cost ≈ cost_TA
    end
end
