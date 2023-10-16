# Taija meets Supercomputing

**Abstract**: [Taija](https://github.com/JuliaTrustworthyAI) is a growing library of packages geared towards *T*rustworthy *A*rtificial *I*ntelligence in *J*uli*a*.  Various ongoing efforts towards making artificial intelligence more trustworthy have one thing in common: they typically increase the overall computational burden involved in training and using machine learning models. To overcome this burden we can either write more performant software or maximise the utilization of existing hardware through parallelization ... or we can do both! This talk will discuss [Taija](https://github.com/JuliaTrustworthyAI)'s recent venture into supercomputing: challenges, success stories and plans for the future. 

## Description

In the wake of recent rapid advances in artificial intelligence (AI), it is more crucial than ever that we take efforts to ensure that the technologies we deploy are trustworthy. Efforts surrounding [Taija](https://github.com/JuliaTrustworthyAI) have so far centered around explainability and uncertainty quantification for supervised machine learning models. [CounterfactualExplanations.jl](https://github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl), for example, is a comprehensive package for generating counterfactual explanations for models trained in [Flux.jl](https://fluxml.ai/Flux.jl/dev/), [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/) and more. 

### 🌐 Why supercomputing?

In practice, we are often required to generate many explanations for many individuals. A firm that is using a machine learning model to screen out job applicants, for example, might be required to explain to each unsuccessful applicant why they were not admitted to the interview stage. In a different context, researchers may need to generate many explanations for [evaluation](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/stable/tutorials/evaluation/) and [benchmarking](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/stable/tutorials/benchmarking/) purposes. In both cases, the involved computational tasks can be parallelized through multi-threading or distributed computing. For this purpose, we have recently added native support for [parallelization](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/stable/tutorials/parallelization/) to our package. This enables more efficient utilization of resources even when working on a personal device. It also paves the way for explaining machine learning models on supercomputers like [DelftBlue](https://www.tudelft.nl/dhpc/system).

### 🤔 How supercomputing?

Our goal has been to minimize the burden on users by facilitating different forms of parallelization through a simple macro. To multi-process the evaluation of a large set of `counterfactuals` using the [MPI.jl](https://juliaparallel.org/MPI.jl/latest/) backend, for example, users just need to 1) load the backend and instantiate the `MPIParallelizer`,

```julia
import MPI
MPI.Init()
parallelizer = MPIParallelizer(MPI.COMM_WORLD)
```

and 2) use the `@with_parallelizer` followed by the `parallelizer` object and the standard API call to evaluate counterfactuals:

```julia
@with_parallelizer parallelizer evaluate(counterfactuals)
```

Under the hood we ...

### 🎯 What is next?

Parallelization is also useful for other [Taija](https://github.com/JuliaTrustworthyAI) packages ...

### 👥 Who is this talk for?

This talk should be useful for anyone interested in either Trustworthy AI or parallel computing or both. 

## Notes