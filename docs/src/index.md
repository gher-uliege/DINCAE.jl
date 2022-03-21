# DINCAE.jl


DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to
reconstruct missing data in satellite observations. It can work with gridded data
(`DINCAE.reconstruct`) or a clouds of points (`DINCAE.reconstruct_points`).
In the later case, the data can be organized in e.g. tracks (or not).

The code is available at:
[https://github.com/gher-ulg/DINCAE.jl](https://github.com/gher-ulg/DINCAE.jl)

The method is described in the following articles:

* Barth, A., Alvera-Azcárate, A., Licer, M., & Beckers, J.-M. (2020). DINCAE 1.0: a convolutional neural network with error estimates to reconstruct sea surface temperature satellite observations. Geoscientific Model Development, 13(3), 1609–1622. https://doi.org/10.5194/gmd-13-1609-2020
* Barth, A., Alvera-Azcárate, A., Troupin, C., & Beckers, J.-M. (2022). DINCAE 2.0: multivariate convolutional neural network with error estimates to reconstruct sea surface temperature satellite and altimetry observations. Geoscientific Model Development, 15(5), 2183–2196. https://doi.org/10.5194/gmd-15-2183-2022

The neural network will be trained on the GPU. Note convolutional neural networks can require a lot of GPU memory depending on the domain size. 
So far, only NVIDIA GPUs are supported by the neural network framework Knet.jl using in DINCAE (beside training on the CPU but which prohibitively slow).


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

