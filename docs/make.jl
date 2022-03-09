using Documenter, DINCAE

makedocs(;
    modules=[DINCAE],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/gher-ulg/DINCAE.jl/blob/{commit}{path}#L{line}",
    sitename="DINCAE.jl",
    authors="Alexander Barth <barth.alexander@gmail.com>",
)

deploydocs(;
    repo="github.com/gher-ulg/DINCAE.jl",
)
