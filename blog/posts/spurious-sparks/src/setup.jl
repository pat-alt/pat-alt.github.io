# Set up:
BLOG_DIR = "blog/posts/spurious-sparks"
using Pkg;
Pkg.activate(BLOG_DIR);

# Packages:
using CSV
using ChainPlots
using DataFrames
using Dates
using Flux
using GLM
using LinearAlgebra
using Metal
using Plots
using Plots.PlotMeasures
using Printf
using TidierData
using Random
using Serialization
using RegressionTables
Random.seed!(2024)

# Utility functions:
include("utils.jl")

RESULTS_DIR = joinpath(BLOG_DIR, "results")
if !isdir(RESULTS_DIR)
    mkdir(RESULTS_DIR)
end
FIGURE_DIR = joinpath(RESULTS_DIR, "figures")