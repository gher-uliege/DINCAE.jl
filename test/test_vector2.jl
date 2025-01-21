# TODO:
# add sin/cos of direction to xin
# maybe as auxiliary variables?

using Test
using DINCAE
using CUDA
using AMDGPU

T = Float32
sz = (100,120)
batch_size = 3

xrec = randn(T,sz...,5,batch_size)

nsites = 3
xtrue = randn(T,sz...,2*nsites,batch_size)

# direction in degress from 0 to 360
directionobs = 360 * rand(T,sz...,nsites,batch_size)

truth_uncertain = false


cost = @time DINCAE.vector2_costfun(xrec,xtrue,truth_uncertain,directionobs)

@test cost isa Number

device_list = Any[identity] # CPU

if CUDA.functional()
    push!(device_list,cu)
end

if AMDGPU.functional()
    push!(device_list,roc)
end

for device in device_list[2:end]
    cost_TA = DINCAE.vector2_costfun(device(xrec),device(xtrue),truth_uncertain,device(directionobs))
    @test cost ≈ cost_TA rtol=1e-4
end
