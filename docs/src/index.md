# DINCAE.jl


DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to
reconstruct missing data in satellite observations. It can work with gridded data
(`DINCAE.reconstruct`) or a clouds of points (`DINCAE.reconstruct_points`).
In the later case, the data can be organized in e.g. tracks (or not).

The code is available at:
[https://github.com/gher-ulg/DINCAE.jl](https://github.com/gher-ulg/DINCAE.jl)


The first version of the method is described in the following paper:

Barth, A., Alvera-Azcárate, A., Licer, M., and Beckers, J.-M.: DINCAE 1.0: a convolutional neural network with error estimates to reconstruct sea surface temperature satellite observations, Geosci. Model Dev., 13, 1609–1622, [https://doi.org/10.5194/gmd-13-1609-2020](https://doi.org/10.5194/gmd-13-1609-2020) , 2020.

The neural network will be trained on the GPU. Note convolutional neural networks can require a lot of GPU memory depending on the domain size. 
So far, only NVIDIA GPUs are supported by the neural network framework Knet.jl using in DINCAE (beside training on the CPU but which prohibitively slow).


## Installation

After installing Julia (available at [https://julialang.org/download](https://julialang.org/download)), one can execute the following Julia command to install the current master version:

```julia
using Pkg
Pkg.add(url="https://github.com/gher-ulg/DINCAE.jl", rev="master")
```

`DINCAE.jl` depends on `Knet.jl` and `CUDA.jl` which will automatically installed. More information is available at [https://denizyuret.github.io/Knet.jl/latest/install/](https://denizyuret.github.io/Knet.jl/latest/install/) and [https://juliagpu.gitlab.io/CUDA.jl/installation/overview/](https://juliagpu.gitlab.io/CUDA.jl/installation/overview/).

After this, you should be able to load `DINCAE` with:

``` julia
using DINCAE
```

## User API


In most cases, a user does only need to interact with the function `DINCAE.reconstruct` or `DINCAE.reconstruct_points`.

```@docs
DINCAE.reconstruct
DINCAE.reconstruct_points
```


## Internal functions

```@docs
DINCAE.load_gridded_nc
DINCAE.NCData
```

