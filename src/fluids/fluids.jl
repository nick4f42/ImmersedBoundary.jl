"""
    Fluids

Properties and discretization of the problem's fluids.
"""
module Fluids

using Parameters

using ..ImmersedBoundary.Dynamics
import ..ImmersedBoundary: discretize

export AbstractFluid, AbstractFluidDomain, FluidDiscretization
export FluidDomain, UniformGrid, MultiLevel, MultiUniformGrid
export frameof, domainof, gridstep, xycoords, baselevel, subdomain, nlevels

abstract type FluidDiscretization end
abstract type AbstractFluidDomain{D<:FluidDiscretization,F<:AbstractFrame} end
abstract type AbstractFluid{D<:AbstractFluidDomain} end

"""
    frameof(domain::AbstractFluidDomain)

The [`AbstractFrame`](@ref) that corresponds to a domain.
"""
function frameof end

"""
    domainof(fluid::AbstractFluid)

The [`AbstractFluidDomain`](@ref) that corresponds to a fluid.
"""
function domainof end

"""
    discretize(fluid::AbstractFluid)
    discretize(domain::AbstractFluidDomain)

The [`FluidDiscretization`](@ref) that corresponds to a fluid or domain.
"""
discretize(fluid::AbstractFluid) = (discretize ∘ domainof)(fluid)

struct FluidDomain{D,F} <: AbstractFluidDomain{D,F}
    discrete::D
    frame::F
end

FluidDomain(discrete) = FluidDomain(discrete, LabFrame())

discretize(domain::FluidDomain) = domain.discrete
frameof(domain::FluidDomain) = domain.frame

struct UniformGrid <: FluidDiscretization
    xs::LinRange{Float64,Int} # x coordinates
    ys::LinRange{Float64,Int} # y coordinates
    h::Float64 # Grid cell size
end

function UniformGrid(h::Float64, lims::Vararg{NTuple{2},2})
    xs, ys = (x0:h:x1 for (x0, x1) in lims)
    return UniformGrid(xs, ys, h)
end

xycoords(grid::UniformGrid) = (grid.xs, grid.ys)

"""
    scale(grid::UniformGrid, k)

Scale a uniform grid about its center by a factor `k`.
"""
function scale(grid::UniformGrid, k)
    xs, ys = map(xycoords(grid)) do r
        c = sum(extrema(r)) / 2
        @. k * (r - c) + c
    end
    return UniformGrid(xs, ys, k * grid.h)
end

"""
    gridstep(grid::UniformGrid)

The grid cell spacing.
"""
gridstep(grid::UniformGrid) = grid.h

struct MultiLevel{D<:FluidDiscretization} <: FluidDiscretization
    base::D
    nlevels::Int
end

const MultiUniformGrid = MultiLevel{UniformGrid}

baselevel(discrete::MultiLevel) = discrete.base
nlevels(discrete::MultiLevel) = discrete.nlevels
subdomain(discrete::MultiLevel, lev::Int) = scale(discrete.base, 2.0^(lev - 1))

"""
    gridstep(domain::MultiLevel, lev::Int)

The grid step of the `lev`th level.
"""
gridstep(discrete::MultiLevel, lev::Int) = gridstep(discrete.base) * 2.0^(lev - 1)

end # module
