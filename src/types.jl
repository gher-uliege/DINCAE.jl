


# upsampling
struct Upsample{TA,N}
    w::TA
    padding::NTuple{N,Int}
end
struct CatSkip
    inner
end
export CatSkip


struct SumSkip
    inner
end
export SumSkip


struct ModelVector2_1
    chain
    truth_uncertain
    gamma
    directionobs
end


