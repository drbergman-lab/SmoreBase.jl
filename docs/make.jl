using SmoreBase
using Documenter

DocMeta.setdocmeta!(SmoreBase, :DocTestSetup, :(using SmoreBase); recursive=true)

makedocs(;
    modules=[SmoreBase],
    authors="Daniel Bergman <danielrbergman@gmail.com> and contributors",
    sitename="SmoreBase.jl",
    format=Documenter.HTML(;
        canonical="https://drbergman-lab.github.io/SmoreBase.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Custom CM Data Types" => "custom_data.md",
        "Plotting" => "plotting.md",
    ],
)

deploydocs(;
    repo="github.com/drbergman-lab/SmoreBase.jl",
    devbranch="main",
)
