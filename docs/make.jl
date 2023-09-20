using Documenter, DINCAE

makedocs(;
    modules=[DINCAE],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/gher-uliege/DINCAE.jl/blob/{commit}{path}#L{line}",
    sitename="DINCAE.jl",
    authors="Alexander Barth <barth.alexander@gmail.com>",
    checkdocs=:none,
)

deploydocs(;
    repo="github.com/gher-uliege/DINCAE.jl",
)
