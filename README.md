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

* Barth, A., Alvera-Azcárate, A., Licer, M., & Beckers, J.-M. (2020). DINCAE 1.0: a convolutional neural network with error estimates to reconstruct sea surface temperature satellite observations. Geoscientific Model Development, 13(3), 1609–1622. https://doi.org/10.5194/gmd-13-1609-2020
* Barth, A., Alvera-Azcárate, A., Troupin, C., & Beckers, J.-M. (2022). DINCAE 2.0: multivariate convolutional neural network with error estimates to reconstruct sea surface temperature satellite and altimetry observations. Geoscientific Model Development, 15(5), 2183–2196. https://doi.org/10.5194/gmd-15-2183-2022

(click [here](CITATION.bib) for the BibTeX entry).

![](examples/Fig/data-avg_2001-09-12.png)
Panel (a) is the original data where we have added clouds (panel (b)). The reconstuction based on the data in panel (b) is shown in panel (c) together
with its expected standard deviation error (panel (d))

DINCAE is indented to be used with a [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) with [CUDA](https://en.wikipedia.org/wiki/CUDA) support (NVIDIA GPU). The code can also run on a [CPU](https://en.wikipedia.org/wiki/Central_processing_unit) but which will be quite slow.

## Installation

You need [Julia](https://julialang.org/downloads) (version 1.7 or later) to run `DINCAE`. The command line interface of Julia is sufficient for `DINCAE`.
If you are using Linux (on a x86_64 CPU), installing and running Julia 1.7.2 is as easy as running these shell commands:

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

`DINCAE.jl` depends on `Knet.jl` and `CUDA.jl` which will automatically be installed. More information is available at [https://denizyuret.github.io/Knet.jl/latest/install/](https://denizyuret.github.io/Knet.jl/latest/install/) and [https://cuda.juliagpu.org/stable/](https://cuda.juliagpu.org/stable/).

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

## Publications

* Barth, A., Alvera-Azcárate, A., Licer, M., & Beckers, J.-M. (2020). DINCAE 1.0: a convolutional neural network with error estimates to reconstruct sea surface temperature satellite observations. Geoscientific Model Development, 13(3), 1609–1622. https://doi.org/10.5194/gmd-13-1609-2020
* Han, Z., He, Y., Liu, G., & Perrie, W. (2020). Application of DINCAE to Reconstruct the Gaps in Chlorophyll-a Satellite Observations in the South China Sea and West Philippine Sea. Remote Sensing, 12(3), 480. https://doi.org/10.3390/rs12030480
* Barth, A., Alvera-Azcárate, A., Troupin, C., & Beckers, J.-M. (2022). DINCAE 2.0: multivariate convolutional neural network with error estimates to reconstruct sea surface temperature satellite and altimetry observations. Geoscientific Model Development, 15(5), 2183–2196. https://doi.org/10.5194/gmd-15-2183-2022
* Jung, S., Yoo, C., & Im, J. (2022). High-Resolution Seamless Daily Sea Surface Temperature Based on Satellite Data Fusion and Machine Learning over Kuroshio Extension. Remote Sensing, 14(3), 575. https://doi.org/10.3390/rs14030575

Thank you for citing relevant previous work in DINCAE if you make a scientific publication.
A bibtex entry can be generated from the DOI by using for example `curl -LH "Accept:  application/x-bibtex"  'https://doi.org/10.5194/gmd-15-2183-2022'`.

Feel free to add your publications by make a pull request or opening an issue.

<!--  LocalWords:  codecov io DINCAE jl Convolutional julia Alvera
 -->
<!--  LocalWords:  Azcárate Licer Beckers convolutional Geosci Dev
 -->
<!--  LocalWords:  Troupin altimetry preprint xzf utils url Knet CUDA
 -->
<!--  LocalWords:  jupyter
 -->
