ENV["CUDA_VISIBLE_DEVICES"]=""

using DINCAE
using Test

@testset "DINCAE.jl" begin
    include("test_DINCAE_SST.jl")
end
