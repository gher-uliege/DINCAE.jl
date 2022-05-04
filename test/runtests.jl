ENV["CUDA_VISIBLE_DEVICES"]=""

using DINCAE
using Test

@testset "reconstruct gridded data" begin
    include("test_DINCAE_SST.jl")
end

@testset "reconstruct point cloud" begin
    include("test_reconstruct_points.jl")
end

@testset "reconstruct vector field" begin
    include("test_vector2.jl")
end
