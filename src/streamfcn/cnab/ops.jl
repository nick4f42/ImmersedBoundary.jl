"""
Compute the matrix (as a LinearMap) that represents the modified Poisson
operator (I + dt/2 * Beta * RC) arising from the implicit treatment of the
Laplacian. A system involving this matrix is solved to compute a trial
circulation that doesn't satisfy the BCs, and then again to use the surface
stresses to update the trial circulation so that it satisfies the BCs

Construct LinearMap to solve
    (I + dt/2 * Beta * RC) * x = b for x
where Beta = 1/(Re * h^2)

Solve by transforming to and from Fourier space and scaling by evals
"""
function A_Ainv_linearmaps(prob::Problem, lap_inv::LaplacianInv, lev::Int)
    fluid = fluidof(prob)
    domain = domainof(fluid)
    grid = discretize(domain)

    hc = gridstep(grid, lev)
    dt = timestep(prob)
    Re = fluid.Re

    Λ = lap_inv.Λ
    dst_plan = lap_inv.dst_plan

    Λexpl = @. inv(1 - Λ * dt / (2 * Re * hc^2)) # Explicit eigenvalues
    A = lap_inv_linearmap(domain, dst_plan, Λexpl)

    Λimpl = @. 1 + Λ * dt / (2 * Re * hc^2) # Implicit eigenvalues
    Ainv = lap_inv_linearmap(domain, dst_plan, Λimpl)

    return (A, Ainv)
end

function A_Ainv_linearmaps(prob::Problem, lap_inv::LaplacianInv)
    nlevel = (nlevels ∘ discretize ∘ fluidof)(prob)

    x = map(lev -> A_Ainv_linearmaps(prob, lap_inv, lev), 1:nlevel)
    As = [A for (A, _) in x]
    Ainvs = [Ainv for (_, Ainv) in x]
    return (As, Ainvs)
end

@with_kw mutable struct B_Times{V<:Vort2Flux,ME,MC,MAinv}
    vort2flux::V
    Ainv::MAinv
    C::MC
    E::ME
    Γtmp::Vector{Float64}
    qtmp::Vector{Float64}
    Γ::Matrix{Float64}
    ψ::Matrix{Float64}
    q::Matrix{Float64}
end

function B_Times(domain::StreamFcnGrid, vort2flux::Vort2Flux, Γtmp, qtmp; Ainv, C, E)
    nlevel = (nlevels ∘ discretize)(domain)
    @unpack nΓ, nq = domain

    # TODO: Move allocations to solver
    Γ = zeros(nΓ, nlevel) # Working array for circulation
    ψ = zeros(nΓ, nlevel) # Working array for streamfunction
    q = zeros(nq, nlevel) # Working array for velocity flux

    return B_Times(vort2flux, Ainv, C, E, Γtmp, qtmp, Γ, ψ, q)
end

function B_linearmap(prob::Problem, B_times::B_Times)
    nftot = 2 * (npoints ∘ bodygroup)(prob)
    return LinearMap(B_times, nftot; issymmetric=true)
end

"""
Precompute 'Binv' matrix by evaluating mat-vec products for unit vectors

This is a big speedup when the interpolation operator E isn't going to
change (no FSI, for instance)
"""
function Binv_linearmap(
    prob::Problem{<:StreamFcnFluid{<:StreamFcnGrid{LabFrame}}}, B
)
    nftot = 2 * (npoints ∘ bodygroup)(prob)

    # Pre-allocate arrays
    Bmat = zeros(nftot, nftot)
    e = zeros(nftot) # Unit vector

    for j in 1:nftot
        # Construct unit vector
        e[max(1, j - 1)] = 0
        e[j] = 1

        @views mul!(Bmat[:, j], B, e)
    end

    Binv = inv(Bmat)
    return LinearMap((y, x) -> mul!(y, Binv, x), nftot; issymmetric=true)

    # TODO: Diagnose why cholesky decomposition leads to non-hermitian error
    # B_decomp = cholesky!(Bmat)
    # return LinearMap((y, x) -> ldiv!(y, B_decomp, x), nftot; issymmetric=true)
end

function Binv_linearmap(prob::Problem, B)
    nftot = 2 * (npoints ∘ bodygroup)(prob)

    # solves f = B*g for g... so g = Binv * f
    # TODO: Add external interface for cg! options
    Binv = LinearMap(nftot; issymmetric=true) do f, g
        cg!(f, B, g; maxiter=5000, reltol=1e-12)
    end

    return Binv
end

"""
Performs one matrix multiply of B*z, where B is the matrix used to solve
for the surface stresses that enforce the no-slip boundary condition.
(B arises from an LU factorization of the full system)

MultiGrid version:
Note ψ is just a dummy variable for computing velocity flux
    Also this only uses Ainv on the first level
"""
function (mem::B_Times)(x::AbstractVector, z::AbstractVector)
    @unpack Ainv, C, E, Γtmp, qtmp, Γ, ψ, q = mem
    vort2flux! = mem.vort2flux

    Γ .= 0

    # Get circulation from surface stress
    # Γ[:, 1] = Ainv * (E * C)' * z
    # Γ = ∇ x (E'*fb)
    mul!(qtmp, E', z)
    mul!(Γtmp, C', qtmp)
    mul!(view(Γ, :, 1), Ainv, Γtmp)

    # Get vel flux from circulation
    vort2flux!(ψ, q, Γ)

    # Interpolate onto the body
    @views mul!(x, E, q[:, 1])
end

@with_kw struct Nonlinear{R<:RhsForce,M}
    rhs_force::R
    C::M
    fq::Vector{Float64}
end

function (mem::Nonlinear)(
    nonlin::AbstractVector, qty::StreamFcnQuantities, Γbc::AbstractVector, lev::Int
)
    grid = discretize(mem.rhs_force.domain)
    @unpack C, fq = mem

    # Get flux-circulation product
    mem.rhs_force(fq, qty, Γbc, lev)

    # Divergence of flux-circulation product
    mul!(nonlin, C', fq)

    # Scaling: 1/hc^2 to convert circulation to vorticity
    hc = gridstep(grid, lev) # Coarse grid spacing
    nonlin .*= 1 / hc^2

    return nothing
end
