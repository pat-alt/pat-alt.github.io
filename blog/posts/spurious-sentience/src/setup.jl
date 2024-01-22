# Set up:
BLOG_DIR = "blog/posts/spurious-sentience"
using Pkg;
Pkg.activate(BLOG_DIR);

# Packages:
using CSV
using ChainPlots
using DataFrames
using Dates
using Flux
# using GLM
using LinearAlgebra
using Metal
using Plots
using Plots.PlotMeasures
using Printf
using TidierData
using Random
using Serialization
# using RegressionTables
Random.seed!(2023)

# Utility functions:
include("utils.jl")