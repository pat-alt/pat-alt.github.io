# Taija meets Supercomputing

**Abstract**: [Taija](https://github.com/JuliaTrustworthyAI) is a growing library of packages geared towards *T*rustworthy *A*rtificial *I*ntelligence in *J*uli*a*.  Various ongoing efforts towards making artificial intelligence more trustworthy have one thing in common: they typically increase the overall computational burden involved in training and using machine learning models. To overcome this burden we can either write more performant software or maximise the utilization of existing hardware through parallelization ... or we can do both! This talk will discuss [Taija](https://github.com/JuliaTrustworthyAI)'s recent venture into supercomputing: challenges, success stories and plans for the future. 

## Description

In the wake of recent rapid advances in artificial intelligence (AI), it is more crucial than ever that we take efforts to ensure that the technologies we deploy are trustworthy. Efforts surrounding [Taija](https://github.com/JuliaTrustworthyAI) have so far centered around explainability and uncertainty quantification for supervised machine learning models. [CounterfactualExplanations.jl](https://github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl), for example, is a comprehensive package for generating counterfactual explanations for models trained in [Flux.jl](https://fluxml.ai/Flux.jl/dev/), [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/) and more. 

### 🌐 Why supercomputing?

In practice, we are often required to generate many explanations for many individuals. A firm that is using a machine learning model to screen out job applicants, for example, might be required to explain to each unsuccessful applicant why they were not admitted to the interview stage. In a different context, researchers may need to generate many explanations for [evaluation](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/stable/tutorials/evaluation/) and [benchmarking](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/stable/tutorials/benchmarking/) purposes. In both cases, the involved computational tasks can be parallelized through multi-threading or distributed computing. For this purpose, we have recently added native support for [parallelization](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/stable/tutorials/parallelization/) to our package. This enables more efficient utilization of resources even when working on a personal device. It also paves the way for explaining machine learning models on supercomputers like [DelftBlue](https://www.tudelft.nl/dhpc/system).

### 🤔 How supercomputing?

Our goal has been to minimize the burden on users by facilitating different forms of parallelization through a simple macro. To multi-process the evaluation of a large set of `counterfactuals` using the [MPI.jl](https://juliaparallel.org/MPI.jl/latest/) backend, for example, users can proceed as follows: firstly, load the backend and instantiate the `MPIParallelizer`,

```julia
import MPI
MPI.Init()
parallelizer = MPIParallelizer(MPI.COMM_WORLD)
```

and then just use the `@with_parallelizer` followed by the `parallelizer` object and the standard API call to evaluate counterfactuals:

```julia
@with_parallelizer parallelizer evaluate(counterfactuals)
```

Under the hood we use standard [MPI.jl](https://juliaparallel.org/MPI.jl/latest/) routines for distributed computing. To avoid depending on [MPI.jl](https://juliaparallel.org/MPI.jl/latest/) we use [package extensions](https://www.youtube.com/watch?v=TiIZlQhFzyk). Similarly, the `ThreadsParallelizer` can be used for multi-threading where we rely on `Base.Threads` routines. It is also possible to combine both forms of parallelization by setting the `threaded` keyword argument of the `MPIParallelizer` to `true`. 

### 🏅 Benchmarking Counterfactuals (case study)

To illustrate how this new functionality can be used in practice, the talk will include a brief case study of our own experience of generating large benchmarks for counterfactual explanations on a supercomputer. This will include some discussion of challenges we encountered along the way and the solutions we have come up with. 

### 🎯 What is next?

Parallelization is also useful for other [Taija](https://github.com/JuliaTrustworthyAI) packages. For example, some of the methods for predictive uncertainty quantification used by [ConformalPrediction.jl](https://github.com/JuliaTrustworthyAI/ConformalPrediction.jl) rely on repeated model training and prediction. This is currently done sequentially and an obvious place for parallelization. To standardize the parallelization API across packages in the [Taija](https://github.com/JuliaTrustworthyAI) ecosystem, we therefore plan to move this functionality into a meta package, TaijaParallelization.jl, much like we have recently moved all plotting functionality into [TaijaPlotting.jl](https://github.com/JuliaTrustworthyAI/TaijaPlotting.jl).

### 👥 Who is this talk for?

This talk should be useful for anyone interested in either Trustworthy AI or parallel computing or both. We are no experts in parallel computing so the level of this talk should also be appropriate for beginners. 

## Notes