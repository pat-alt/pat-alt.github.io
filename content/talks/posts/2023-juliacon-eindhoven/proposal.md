# Taija meets Supercomputing

**Abstract**: [Taija](https://github.com/JuliaTrustworthyAI) is a growing library of packages geared towards *T*rustworthy *A*rtificial *I*ntelligence in *J*uli*a*.  Various ongoing efforts towards making artificial intelligence more trustworthy have one thing in common: they typically increase the overall computational burden involved in training and using machine learning models. To overcome this burden we can either write more performant software or maximise the utilization of existing hardware through parallelization ... or we can do both! This talk will discuss [Taija](https://github.com/JuliaTrustworthyAI)'s recent venture into supercomputing: challenges, success stories and plans for the future. 

## Description

In the wake of recent rapid advances in artificial intelligence (AI), it is more crucial than ever that we take efforts to ensure that the technologies we deploy are trustworthy. Efforts surrounding [Taija](https://github.com/JuliaTrustworthyAI) have so far centered around explainability and uncertainty quantification for supervised machine learning models. [CounterfactualExplanations.jl](https://github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl), for example, is a comprehensive package for generating counterfactual explanations for models trained in [Flux.jl](https://fluxml.ai/Flux.jl/dev/), [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/) and more. 

### 🌐 Why supercomputing?

In practice, we are often required to generate many explanations for many individuals. A firm that is using a machine learning model to screen out job applicants, for example, might be required to explain to each unsuccessful applicant why they were not admitted to the interview stage. In a different context, researchers may need to generate many explanations for evaluation and benchmarking purposes. In both cases, the task of generating explanations can be parallelized through multi-threading or distributed computing. 

### 🤔 How supercomputing?

### 🎯 What is next?

### 👥 Who is this talk for?

## Notes