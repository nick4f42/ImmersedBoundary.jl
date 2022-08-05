abstract type StreamFcnDomain{D,F} <: AbstractFluidDomain{D,F} end

struct StreamFcnGrid{F<:AbstractFrame} <: StreamFcnDomain{MultiUniformGrid,F}
    discrete::MultiUniformGrid
    frame::F
    nx::Int # Number of x grid cells
    ny::Int # Number of y grid cells
    nu::Int # Number of x flux points
    nq::Int # Number of flux points
    nΓ::Int # Number of circulation points
    # Boundary condition index offsets
    L::Int # Left
    R::Int # Right
    B::Int # Bottom
    T::Int # Top
end

discretize(grid::StreamFcnGrid) = grid.discrete
frameof(grid::StreamFcnGrid) = grid.frame

StreamFcnDomain(discrete::FluidDiscretization) = StreamFcnDomain(FluidDomain(discrete))

function StreamFcnDomain(domain::FluidDomain{UniformGrid})
    d = FluidDomain(MultiLevel(domain.discrete, 1), domain.frame)
    return StreamFcnDomain(d)
end

function StreamFcnDomain(domain::FluidDomain)
    @unpack discrete, frame = domain
    xs, ys = (xycoords ∘ baselevel)(discrete)
    nx = length(xs) - 1
    ny = length(ys) - 1

    nu = (nx + 1) * ny
    nv = nx * (ny + 1)
    nq = nu + nv
    nΓ = (nx - 1) * (ny - 1)

    L = 0
    R = ny + 1
    B = 2 * (ny + 1)
    T = 2 * (ny + 1) + nx + 1

    return StreamFcnGrid(discrete, frame, nx, ny, nu, nq, nΓ, L, R, B, T)
end

# TODO: Add optimizations for time-invariant freestream

struct StreamFcnFluid{D<:StreamFcnDomain} <: AbstractFluid{D}
    domain::D
    freestream::FuncWrap{SVector{2,Float64},Tuple{Float64}} # t -> (ux, uy)
    Re::Float64
    function StreamFcnFluid(domain::D, freestream, Re) where {D<:StreamFcnDomain}
        return new{D}(domain, freestream, Re)
    end
end

StreamFcnFluid(domain, args...) = StreamFcnFluid(StreamFcnDomain(domain), args...)

domainof(fluid::StreamFcnFluid) = fluid.domain

xflux_ranges(grid::UniformGrid) = (grid.xs, midpoints(grid.ys))
yflux_ranges(grid::UniformGrid) = (midpoints(grid.xs), grid.ys)
circ_ranges(grid::UniformGrid) = (grid.xs[2:end-1], grid.ys[2:end-1])

innerpoints(r::AbstractVector) = @views r[begin+1:end-1]
midpoints(r::AbstractVector) = @views @. (r[begin:end-1] + r[begin+1:end]) / 2

function flatten_circ(Γ::AbstractArray, domain::StreamFcnGrid)
    nΓ = domain.nΓ
    return reshape(Γ, nΓ, size(Γ)[3:end]...)
end

function unflatten_circ(Γ::AbstractArray, domain::StreamFcnGrid)
    @unpack nx, ny = domain
    return reshape(Γ, nx - 1, ny - 1, size(Γ)[2:end]...)
end

function unflatten_circ(Γ::AbstractMatrix, domain::StreamFcnGrid, lev)
    return unflatten_circ(view(Γ, :, lev), domain)
end

function split_flux(q::AbstractArray, domain::StreamFcnGrid)
    @unpack nx, ny, nu = domain

    uflat = @view q[1:nu, ..]
    vflat = @view q[(nu+1):end, ..]

    dims = size(q)[2:end]

    u = @views reshape(uflat, nx + 1, ny, dims...)
    v = @views reshape(vflat, nx, ny + 1, dims...)
    return (u, v)
end

function split_flux(q::AbstractMatrix, domain::StreamFcnGrid, lev)
    return split_flux(view(q, :, lev), domain)
end
