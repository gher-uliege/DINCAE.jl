[![Build Status](https://github.com/gher-ulg/DINCAE.jl/workflows/CI/badge.svg)](https://github.com/gher-ulg/DINCAE.jl/actions)
[![codecov.io](http://codecov.io/github/gher-ulg/DINCAE.jl/coverage.svg?branch=main)](http://codecov.io/github/gher-ulg/DINCAE.jl?branch=main)
[![documentation stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gher-ulg.github.io/DINCAE.jl/stable/)
[![documentation dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gher-ulg.github.io/DINCAE.jl/dev/)

# DINCAE.jl

DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to reconstruct missing data in satellite observations.
This repository contains the julia port of DINCAE. The original [python code](https://github.com/gher-ulg/DINCAE) is no longer maintained.

Utilities (for plotting and data preparation) are available in a separate repository
https://github.com/gher-ulg/DINCAE_utils.jl

The method is described in the following articles:

* Barth, A., Alvera-Azcárate, A., Licer, M., and Beckers, J.-M.: DINCAE 1.0: a convolutional neural network with error estimates to reconstruct sea surface temperature satellite observations, Geosci. Model Dev., 13, 1609-1622, https://doi.org/10.5194/gmd-13-1609-2020, 2020.
* Barth, A., Alvera-Azcárate, A., Troupin, C., and Beckers, J.-M.: DINCAE 2: multivariate convolutional neural network with error estimates to reconstruct sea surface temperature satellite and altimetry observations, Geosci. Model Dev. Discuss. [preprint], https://doi.org/10.5194/gmd-2021-353, 2022 (accepted).


![](examples/Fig/data-avg_2001-09-12.png)
Panel (a) is the original data where we have added clouds (panel (b)). The reconstuction based on the data in panel (b) is shown in panel (c) together
with its expected standard deviation error (panel (d))

## Installation

You need [Julia](https://julialang.org/downloads) (version 1.7 or later) to run `DINCAE`. The command line interface is sufficient for `DINCAE`.
If you are using Linux, installing and running Julia 1.7.2 is as easy as running these shell commands:

```bash
curl https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.2-linux-x86_64.tar.gz | tar -xzf -
julia-1.7.2/bin/julia
```

This installs Julia in the current directory under the folder `julia-1.7.2`.
For more information and other platforms, please see [platform specific instructions](https://julialang.org/downloads/platform/) for further installation instructions.
You can check the latest available version of Julia at https://julialang.org/downloads but avoid beta releases and release candidates if you are new to Julia.

Inside a Julia terminal, you can download and install `DINCAE` and `DINCAE_utils` by issuing these commands:

```julia
using Pkg
Pkg.add(url="https://github.com/gher-ulg/DINCAE.jl", rev="main")
Pkg.add(url="https://github.com/gher-ulg/DINCAE_utils.jl", rev="main")
```

`DINCAE.jl` depends on `Knet.jl` and `CUDA.jl` which will automatically installed. More information is available at [https://denizyuret.github.io/Knet.jl/latest/install/](https://denizyuret.github.io/Knet.jl/latest/install/) and [https://cuda.juliagpu.org/stable/](https://cuda.juliagpu.org/stable/).

After this, you should be able to load `DINCAE` with:

``` julia
using DINCAE
```


### Updating DINCAE

To update `DINCAE`, run the following command and restart Julia (or restart the jupyter notebook kernel using `Kernel` -> `Restart`):

```julia
using Pkg
Pkg.update("DINCAE")
```

Note that Julia does not directly delete the previous installed version.
To check if you have the latest version run the following command:

```julia
using Pkg
Pkg.status()
```

The latest version number is available from [here](https://github.com/gher-ulg/DINCAE.jl/releases).

## Documentation

More information is available in the [documentation](https://gher-ulg.github.io/DINCAE.jl/stable/) and the tutorial (available as
[script](https://github.com/gher-ulg/DINCAE.jl/blob/main/examples/DINCAE_tutorial.jl) and [jupyter notebook](https://github.com/gher-ulg/DINCAE.jl/blob/main/examples/DINCAE_tutorial.ipynb)).

<!--  LocalWords:  codecov io DINCAE jl Convolutional julia Alvera
 -->
<!--  LocalWords:  Azcárate Licer Beckers convolutional Geosci Dev
 -->
<!--  LocalWords:  Troupin altimetry preprint xzf utils url Knet CUDA
 -->
<!--  LocalWords:  jupyter
 -->