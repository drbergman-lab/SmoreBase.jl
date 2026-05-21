using SMoReBase
using Documenter

DocMeta.setdocmeta!(SMoReBase, :DocTestSetup, :(using SMoReBase); recursive=true)

makedocs(;
    modules=[SMoReBase],
    authors="Daniel Bergman <danielrbergman@gmail.com> and contributors",
    sitename="SMoReBase.jl",
    format=Documenter.HTML(;
        canonical="https://drbergman-lab.github.io/SMoReBase.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/drbergman-lab/SMoReBase.jl",
    devbranch="main",
)
