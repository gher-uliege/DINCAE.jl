# DINCAE: Data-Interpolating Convolutional Auto-Encoder
# Copyright (C) 2019 Alexander Barth
#
# This file is part of DINCAE.

# DINCAE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# DINCAE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with DINCAE. If not, see <http://www.gnu.org/licenses/>.


"""
DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to
reconstruct missing data in satellite observations.

For most applications it is sufficient to call the function
`DINCAE.reconstruct` directly.

The code is available at:
[https://github.com/gher-uliege/DINCAE.jl](https://github.com/gher-uliege/DINCAE.jl)
"""
module DINCAE
using Base.Threads
using CUDA
using Dates
using Random
using NCDatasets
using Printf
using Statistics
using Knet
using ThreadsX

import Base: length
import Base: size
import Base: getindex
import Random: shuffle!

include("data.jl")
include("model.jl")
include("points.jl")
end
