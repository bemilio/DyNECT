using Plots
using CSV
using DataFrames
include("overtake_utils.jl")
plt = plot(; aspect_ratio=:equal,
            xlim=(-10,10),
            ylim=(-10,20)
        )
drawRectangle!(plt, 0., 0., pi/2, 20., 10., :blue)
display(plt)
readline()
