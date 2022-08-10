struct LaplacianInv{D<:StreamFcnGrid}
    domain::D
    b_temp::Matrix{Float64}
    x_temp::Matrix{Float64}
    Λ::Matrix{Float64}
    dst_plan::FFTW.r2rFFTWPlan{Float64,(7, 7),false,2,Vector{Int64}}
    work::Matrix{Float64}
    scale::Float64
    nΓ::Int
end

function lap_inv_linearmap(lap_inv::LaplacianInv)
    nΓ = length(lap_inv.Λ)
    return LinearMap(lap_inv, nΓ; issymmetric=true)
end

lap_inv_linearmap(args...) = lap_inv_linearmap(LaplacianInv(args...))

function LaplacianInv(domain::StreamFcnGrid)
    @unpack nx, ny = domain

    Λ = lap_eigs(domain)

    b = ones(nx - 1, ny - 1)
    dst_plan = FFTW.plan_r2r(b, FFTW.RODFT00, [1, 2]; flags=FFTW.EXHAUSTIVE)

    return LaplacianInv(domain, dst_plan, Λ)
end

function LaplacianInv(domain::StreamFcnGrid, dst_plan, Λ)
    @unpack nx, ny = domain

    # TODO: Test without x_temp b_temp intermediaries
    b_temp = zeros(nx - 1, ny - 1) # Input
    x_temp = zeros(nx - 1, ny - 1) # Output
    work = zeros(nx - 1, ny - 1)
    scale = 1 / (4 * nx * ny)

    return LaplacianInv(domain, b_temp, x_temp, Λ, dst_plan, work, scale, domain.nΓ)
end

function (mem::LaplacianInv)(x::AbstractVector, b::AbstractVector)
    @unpack domain, b_temp, x_temp, Λ, dst_plan, work, scale = mem
    @unpack nx, ny = domain

    # TODO: Test if b_temp and x_temp are beneficial
    b_temp .= reshape(b, size(b_temp))

    mul!(work, dst_plan, b_temp)
    @. work *= scale / Λ
    mul!(x_temp, dst_plan, work)

    x .= reshape(x_temp, size(x))

    return nothing
end

function lap_eigs(domain::StreamFcnGrid)
    @unpack nx, ny = domain
    return @. -2 * (cos(π * (1:(nx-1)) / nx) + cos(π * (1:(ny-1)) / ny)' - 2)
end

function C_linearmap(domain::StreamFcnGrid)
    @unpack nq, nΓ = domain

    C(y, x) = curl!(y, x, domain)
    CT(y, x) = rot!(y, x, domain)
    return LinearMap(C, CT, nq, nΓ)
end

"""
    curl!(q, ψ, domain)

Discrete curl operator.

Note that this doesn't actualy get used in the MultiDomain formulation, but it is the
adjoint to `rot`, which does get called.
"""
function curl!(q_flat, ψ_flat, domain::StreamFcnGrid)
    @unpack nx, ny = domain

    ψ = reshape(ψ_flat, nx - 1, ny - 1)
    qx, qy = split_flux(q_flat, domain)

    # x fluxes

    let i = 2:nx, j = 2:ny-1
        @views @. qx[i, j] = ψ[i-1, j] - ψ[i-1, j-1]
    end
    let i = 2:nx, j = 1
        @views @. qx[i, j] = ψ[i.-1, j] # Top boundary
    end
    let i = 2:nx, j = ny
        @views @. qx[i, j] = -ψ[i.-1, j.-1] # Bottom boundary
    end

    # y-fluxes

    let i = 2:nx-1, j = 2:ny
        @views @. qy[i, j] = ψ[i-1, j-1] - ψ[i, j-1]
    end
    let i = 1, j = 2:ny
        @views @. qy[i, j] = -ψ[i, j-1] # Left boundary
    end
    let i = nx, j = 2:ny
        @views @. qy[i, j] = ψ[i-1, j-1] # Right boundary
    end

    return nothing
end

"""
    rot!(Γ, q, domain)

Transpose of discrete curl (R matrix in previous versions of the code).
"""
function rot!(Γ_flat, q_flat, domain::StreamFcnGrid)
    @unpack nx, ny = domain

    Γ = reshape(Γ_flat, nx - 1, ny - 1)
    qx, qy = split_flux(q_flat, domain)

    i = 2:nx
    j = 2:ny

    @views @. Γ[i-1, j-1] = (qx[i, j-1] - qx[i, j]) + (qy[i, j] - qy[i-1, j])

    return nothing
end

"""
    curl!(q, ψ, ψbc, domain)

Compute velocity flux from streamfunction on MultiDomain.

Note: requires streamfunction from coarser grid on edge of current domain (stored in ψ_bc)
"""
function curl!(q_flat, ψ_flat, ψ_bc, domain::StreamFcnGrid)
    @unpack nx, ny, T, B, L, R = domain

    ψ = unflatten_circ(ψ_flat, domain)
    qx, qy = split_flux(q_flat, domain)

    # x fluxes

    let i = 2:nx, j = 2:ny-1
        @views @. qx[i, j] = ψ[i-1, j] - ψ[i-1, j-1]
    end
    let i = 2:nx, j = 1
        @views @. qx[i, j] = ψ[i-1, j] - ψ_bc[B+i] # Bottom boundary
    end
    let i = 2:nx, j = ny
        @views @. qx[i, j] = ψ_bc[i+T] - ψ[i-1, j-1] # Top boundary
    end

    let i = 1, j = 1:ny
        @views @. qx[i, j] = ψ_bc[(L+1)+j] - ψ_bc[L+j] # Left boundary
    end
    let i = nx + 1, j = 1:ny
        @views @. qx[i, j] = ψ_bc[(R+1)+j] - ψ_bc[R+j] # Right boundary
    end

    # y fluxes

    let i = 2:nx-1, j = 2:ny
        @views @. qy[i, j] = ψ[i-1, j-1] - ψ[i, j-1]
    end
    let i = 1, j = 2:ny
        @views @. qy[i, j] = ψ_bc[L+j] - ψ[i, j-1] # Left boundary
    end
    let i = nx, j = 2:ny
        @views @. qy[i, j] = ψ[i-1, j-1] - ψ_bc[R+j] # Right boundary
    end
    let i = 1:nx, j = 1
        @views @. qy[i, j] = ψ_bc[B+i] - ψ_bc[(B+1)+i] # Bottom boundary
    end
    let i = 1:nx, j = ny + 1
        @views @. qy[i, j] = ψ_bc[T+i] - ψ_bc[(T+1)+i] # Top boundary
    end

    return nothing
end

function coarsify!(Γc_flat::AbstractVector, Γ_flat::AbstractVector, domain::StreamFcnGrid)
    @unpack nx, ny = domain
    Γc = unflatten_circ(Γc_flat, domain)
    Γ = unflatten_circ(Γ_flat, domain)

    js = @. ny ÷ 2 .+ ((-ny÷2+2):2:(ny÷2-2))
    jcs = @. ny ÷ 2 .+ ((-ny÷4+1):(ny÷4-1))
    is = @. nx ÷ 2 .+ ((-nx÷2+2):2:(nx÷2-2))
    ics = @. nx ÷ 2 .+ ((-nx÷4+1):(nx÷4-1))

    for (j, jc) in zip(js, jcs), (i, ic) in zip(is, ics)
        Γc[ic, jc] =
            Γ[i, j] +
            0.5 * (Γ[i+1, j] + Γ[i, j+1] + Γ[i-1, j] + Γ[i, j-1]) +
            0.25 * (Γ[i+1, j+1] + Γ[i+1, j-1] + Γ[i-1, j-1] + Γ[i-1, j+1])
    end

    return nothing
end

"""
    get_bc!(rbc, r_flat, domain)

Given vorticity on a larger, coarser mesh, interpolate it's values to the edge of a smaller,
finer mesh.
"""
function get_bc!(rbc::AbstractVector, r_flat::AbstractVector, domain::StreamFcnGrid)
    @unpack nx, ny, T, B, L, R = domain
    r = unflatten_circ(r_flat, domain)

    # Get interpolated boundary conditions on finer grid
    let i = (nx ÷ 4) .+ (0:nx÷2), ibc = 1:2:nx+1
        @views @. rbc[B+ibc] = r[i, ny÷4]
        @views @. rbc[T+ibc] = r[i, 3*ny÷4]
    end

    let i = (nx ÷ 4) .+ (1:nx÷2), ibc = 2:2:nx
        @views @. rbc[B+ibc] = 0.5 * (r[i, ny÷4] + r[i-1, ny÷4])
        @views @. rbc[T+ibc] = 0.5 * (r[i, 3*ny÷4] + r[i-1, 3*ny÷4])
    end

    let j = (ny ÷ 4) .+ (0:ny÷2), jbc = 1:2:ny+1
        @views @. rbc[L+jbc] = r[nx÷4, j]
        @views @. rbc[R+jbc] = r[3*nx÷4, j]
    end

    let j = (ny ÷ 4) .+ (1:ny÷2), jbc = 2:2:ny
        @views @. rbc[L+jbc] = 0.5 * (r[nx÷4, j] + r[nx÷4, j-1])
        @views @. rbc[R+jbc] = 0.5 * (r[3*nx÷4, j] + r[3*nx÷4, j-1])
    end

    # Account for scaling between grids
    rbc .*= 0.25

    return nothing
end

"""
    apply_bc!(r_flat, rbc, fac, domain)

given vorticity at edges of domain, rbc, (from larger, coarser mesh), add values to correct
laplacian of vorticity  on the (smaller, finer) domain, r.

r is a vorticity-like array of size (nx-1)×(ny-1)
"""
function apply_bc!(
    r_flat::AbstractVector, rbc::AbstractVector, fac::Float64, domain::StreamFcnGrid
)
    @unpack nx, ny, T, B, L, R = domain
    r = unflatten_circ(r_flat, domain)

    # add bc's from coarser grid
    @views let i = 1:nx-1
        let j = 1
            @. r[i, j] += fac * rbc[(B+1)+i]
        end
        let j = ny - 1
            @. r[i, j] += fac * rbc[(T+1)+i]
        end
    end

    @views let j = 1:ny-1
        let i = 1
            @. r[i, j] += fac * rbc[(L+1)+j]
        end
        let i = nx - 1
            @. r[i, j] += fac * rbc[(R+1)+j]
        end
    end

    return nothing
end

function avg_flux!(
    Q::AbstractVector, qty::StreamFcnQuantities, domain::StreamFcnGrid, lev::Int
)
    @unpack nx, ny = domain

    qx, qy = split_flux(qty.q, domain, lev)
    q0x, q0y = split_flux(qty.q0, domain, lev)
    Qx, Qy = split_flux(Q, domain)

    # Index into Qx from (1:nx+1)×(2:ny)
    let i = 1:nx+1, j = 2:ny
        @views @. Qx[i, j] = (qx[i, j] + qx[i, j.-1] + q0x[i, j] + q0x[i, j.-1]) / 2
    end

    # Index into Qy from (2:nx)×(1:ny+1)
    let i = 2:nx, j = 1:ny+1
        @views @. Qy[i, j] = (qy[i, j] + qy[i.-1, j] + q0y[i, j] + q0y[i.-1, j]) / 2
    end

    return Q
end

"""
   direct_product!(fq, Q, Γ, Γbc, domain)

Gather the product used in computing advection term

fq is the output array: the product of flux and circulation such that the nonlinear term is
C'*fq (or ∇⋅fq)
"""
function direct_product!(fq, Q, Γ, Γbc, domain::StreamFcnGrid)
    # Zero out in case some locations aren't indexed
    # TODO: Set unindexed locations to zero to avoid blanket set
    fq .= 0

    direct_product_loops!(fq, Q, Γ, Γbc, domain)

    return nothing
end

"""
   direct_product_loops!(fq, Q, Γ, Γbc, domain)

Helper function to compute the product of Q and Γ so that the advective term is ∇⋅fq
"""
function direct_product_loops!(fq, Q, Γ, Γbc, domain::StreamFcnGrid)
    @unpack nx, ny, T, B, L, R = domain

    Qx, Qy = split_flux(Q, domain)
    fqx, fqy = split_flux(fq, domain)

    # x fluxes
    @views let i = 2:nx,
        j = 2:ny-1,
        fqx = fqx[i, j],
        Qy1 = Qy[i, j.+1],
        Γ1 = Γ[i.-1, j],
        Qy2 = Qy[i, j],
        Γ2 = Γ[i.-1, j.-1]

        @. fqx = Qy1 * Γ1 + Qy2 * Γ2
    end

    # x fluxes bottom boundary
    @views let i = 2:nx,
        j = 1,
        fqx = fqx[i, j],
        Qy1 = Qy[i, j.+1],
        Γ1 = Γ[i.-1, j],
        Qy2 = Qy[i, j],
        Γ2 = Γbc[B.+i]

        @. fqx = Qy1 * Γ1 + Qy2 * Γ2
    end

    # x fluxes top boundary
    @views let i = 2:nx,
        j = ny,
        fqx = fqx[i, j],
        Qy1 = Qy[i, j],
        Γ1 = Γ[i.-1, j.-1],
        Qy2 = Qy[i, j.+1],
        Γ2 = Γbc[T.+i]

        @. fqx = Qy1 * Γ1 + Qy2 * Γ2
    end

    # y fluxes
    @views let i = 2:nx-1,
        j = 2:ny,
        fqy = fqy[i, j],
        Qx1 = Qx[i.+1, j],
        Γ1 = Γ[i, j.-1],
        Qx2 = Qx[i, j],
        Γ2 = Γ[i.-1, j.-1]

        @. fqy = -(Qx1 * Γ1 + Qx2 * Γ2)
    end

    # y fluxes left boundary
    @views let i = 1,
        j = 2:ny,
        fqy = fqy[i, j],
        Qx1 = Qx[i.+1, j],
        Γ1 = Γ[i, j.-1],
        Qx2 = Qx[i, j],
        Γ2 = Γbc[L.+j]

        @. fqy = -(Qx1 * Γ1 + Qx2 * Γ2)
    end

    # y fluxes right boundary
    @views let i = nx,
        j = 2:ny,
        fqy = fqy[i, j],
        Qx1 = Qx[i, j],
        Γ1 = Γ[i.-1, j.-1],
        Qx2 = Qx[i.+1, j],
        Γ2 = Γbc[R.+j]

        @. fqy = -(Qx1 * Γ1 + Qx2 * Γ2)
    end

    return nothing
end

function base_flux!(
    qty::StreamFcnQuantities, fluid::StreamFcnFluid{<:StreamFcnGrid{LabFrame}}, t::Float64
)
    grid = discretize(fluid)
    u, v = fluid.freestream(t)
    qx, qy = split_flux(qty.q0, domainof(fluid))

    for lev in 1:nlevels(grid)
        # Coarse grid spacing
        hc = gridstep(grid, lev)

        qx[:, :, lev] .= u * hc
        qy[:, :, lev] .= v * hc
    end
end

# TODO: Implement moving grid base_flux!

@with_kw struct Vort2Flux{G<:StreamFcnGrid,M}
    domain::G
    ψbc::Vector{Float64}
    Γtmp::Vector{Float64}
    Δinv::M
end

"""
    vort2flux!(ψ, q, Γ)

Multiscale method to solve C^T C s = omega and return the velocity, C s.
Results are returned in vel on each of the first nlev grids.

Warning: the vorticity field on all but the finest mesh is modified by the routine in the
following way: the value in the center of the domain is interpolated from the next finer
mesh (the value near the edge is not changed.
"""
(mem::Vort2Flux)(qty::StreamFcnQuantities) = mem(qty.ψ, qty.q, qty.Γ)

function (mem::Vort2Flux)(ψ::AbstractMatrix, q::AbstractMatrix, Γ::AbstractMatrix)
    domain = mem.domain
    grid = discretize(domain)
    nlevel = nlevels(grid)

    @unpack ψbc, Γtmp, Δinv = mem
    @unpack nx, ny = domain

    # Interpolate values from finer grid to center region of coarse grids
    for lev in 2:nlevel
        @views coarsify!(Γ[:, lev], Γ[:, lev-1], domain)
    end

    # Invert Laplacian on largest grid with zero boundary conditions
    ψ .= 0
    ψbc .= 0
    @views mul!(ψ[:, nlevel], Δinv, Γ[:, nlevel]) # Δψ = Γ
    @views curl!(q[:, nlevel], ψ[:, nlevel], ψbc, domain) # q = ∇×ψ

    # Telescope in to finer grids, using boundary conditions from coarser
    for lev in (nlevel-1):-1:1
        @views Γtmp .= Γ[:, lev]
        @views get_bc!(ψbc, ψ[:, lev+1], domain)
        apply_bc!(Γtmp, ψbc, 1.0, domain)

        @views mul!(ψ[:, lev], Δinv, Γtmp) # Δψ = Γ
        if lev < nlevel
            @views curl!(q[:, lev], ψ[:, lev], ψbc, domain) # q = ∇×ψ
        end
    end

    return nothing
end
