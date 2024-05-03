ENV["CUDA_VISIBLE_DEVICES"]=""

using DINCAE
using Test

@testset "reconstruct gridded data" begin
    include("test_DINCAE_SST.jl")
    include("test_regL2.jl")
end

@testset "reconstruct point cloud" begin
    include("test_reconstruct_points.jl")
    include("test_reconstruct_points_laplacian.jl")
    include("test_interp_adjoint.jl")
end

@testset "reconstruct vector field" begin
    include("test_vector2.jl")
end

@testset "utilities" begin
    include("test_utils.jl")
end
