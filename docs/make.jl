using Documenter
using DyNECT

makedocs(
    sitename = "DyNECT.jl",
    modules = [DyNECT],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/bemilio/DyNECT.git",
)
