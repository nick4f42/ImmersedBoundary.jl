struct GetTrialState{F<:StreamFcnFluid{<:StreamFcnGrid},N<:Nonlinear,V<:Vort2Flux,MA,MAinv}
    fluid::F
    scheme::SchemeCNAB
    nonlinear::N
    vort2flux::V
    As::Vector{MA}
    Ainvs::Vector{MAinv}
    rhsbc::Vector{Float64}
    rhs::Vector{Float64}
    bc::Vector{Float64}
end

function GetTrialState(;
    prob::Problem{<:StreamFcnFluid{<:StreamFcnGrid}},
    nonlinear,
    vort2flux,
    As,
    Ainvs,
    rhsbc,
    rhs,
    bc,
)
    fluid = fluidof(prob)
    scheme = schemeof(prob)
    return GetTrialState(fluid, scheme, nonlinear, vort2flux, As, Ainvs, rhsbc, rhs, bc)
end

function (mem::GetTrialState)(qs::AbstractMatrix, Γs::AbstractMatrix, state::State)
    @unpack fluid, scheme = mem
    domain = domainof(fluid)
    grid = discretize(domain)
    nlevel = nlevels(grid)
    dt = timestep(scheme)

    nonlin = state.nonlin
    qty = quantities(state)

    nonlinear!, vort2flux! = mem.nonlinear, mem.vort2flux
    @unpack As, Ainvs, bc, rhs, rhsbc = mem

    bc .= 0
    rhsbc .= 0

    for lev in nlevel:-1:1
        hc = gridstep(grid, lev)

        if lev < nlevel
            @views get_bc!(bc, qty.Γ[:, lev+1], domain)

            fac = 0.25 * dt / (fluid.Re * hc^2)
            apply_bc!(rhsbc, bc, fac, domain)
        end

        # Account for scaling between grids
        # Don't need bc's for anything after this, so we can rescale in place
        bc .*= 0.25
        #compute the nonlinear term for the current time step
        @views nonlinear!(nonlin[1][:, lev], qty, bc, lev)

        @views mul!(rhs, As[lev], qty.Γ[:, lev])

        for n in 1:length(scheme.β)
            @. rhs += dt * scheme.β[n] * (@view nonlin[n][:, lev])
        end

        # Include boundary conditions
        rhs .+= rhsbc

        # Trial circulation  Γs = Ainvs * rhs
        @views mul!(Γs[:, lev], Ainvs[lev], rhs)
    end

    # Store nonlinear solution for use in next time step
    # Cycle nonlinear arrays
    cycle!(nonlin)

    vort2flux!(qty.ψ, qs, Γs)
    return nothing
end

cycle!(a::Vector) = isempty(a) ? a : pushfirst!(a, pop!(a))

"""
Solve the Poisson equation (25) in Colonius & Taira (2008).

Dispatch based on the type of motion in the problem - allows precomputing
regularization and interpolation where possible.
"""
struct BoundaryForcer{B,MBinv,ME}
    bodytype::Type{B}
    Binv::MBinv
    E::ME
    Ftmp::Vector{Float64}
    Q::Vector{Float64}
    h::Float64
end

function BoundaryForcer(;
    prob::Problem{<:StreamFcnFluid{<:StreamFcnGrid}}, Binv, E, Ftmp, Q
)
    h = (gridstep ∘ baselevel ∘ discretize ∘ fluidof)(prob)
    bodytype = (eltype ∘ bodygroup)(prob)
    return BoundaryForcer(bodytype, Binv, E, Ftmp, Q, h)
end

function (mem::BoundaryForcer{RigidBody{GridFrame}})(F̃b, qs, q0, ::BodyGroupPoints)
    # Rigid bodies fixed in the grid frame
    # Solve modified Poisson problem for uB = 0 and bc2 = 0
    # Bf̃ = Eq = ECψ

    @unpack Binv, E, Ftmp, Q = mem

    @. Q = qs + q0

    # E*(qs .+ state.q0)
    mul!(Ftmp, E, Q)
    mul!(F̃b, Binv, Ftmp)

    return nothing
end

function (mem::BoundaryForcer)(F̃b, qs, q0, points::BodyGroupPoints)
    # Bodies moving in the grid frame
    # Solve the Poisson problem for bc2 = 0 (???) with nonzero boundary velocity ub
    # Bf̃ = Eq - ub
    #    = ECψ - ub

    @unpack Binv, E, Ftmp, Q, h = mem

    @. Q = qs + q0

    ub = vec(points.v) # flattened velocities

    mul!(Ftmp, E, Q) # E*(qs .+ state.q0)
    @. Ftmp -= ub * h # Enforce no-slip conditions
    mul!(F̃b, Binv, Ftmp)

    return nothing
end

@with_kw struct CircProjecter{MAinv,MC,ME}
    Ainv::MAinv
    C::MC
    E::ME
    Γtmp::Vector{Float64}
    qtmp::Vector{Float64}
end

"""
    project_circ!(Γs, state, prob)

Update circulation to satisfy no-slip condition.

This allows precomputing regularization and interpolation where possible.
"""
function (mem::CircProjecter)(qty::StreamFcnQuantities, Γs::AbstractMatrix)
    # High-level version:
    #     Γ = Γs - Ainv * (E*C)'*F̃b

    Γs1 = @view Γs[:, 1]
    @unpack Ainv, C, E, Γtmp, qtmp = mem

    qty.Γ .= Γs

    mul!(qtmp, E', view(qty.Fb, :, 1))
    mul!(Γtmp, C', qtmp)
    mul!(Γs1, Ainv, Γtmp) # use Γs as temporary buffer
    @views qty.Γ[:, 1] .-= Γs1

    return nothing
end

struct Solver{
    P<:Problem{<:StreamFcnFluid{<:StreamFcnGrid}},
    R<:Reg,
    G<:GetTrialState,
    B<:BoundaryForcer,
    C<:CircProjecter,
    V<:Vort2Flux,
} <: AbstractSolver
    prob::P
    qs::Matrix{Float64} # Trial flux
    Γs::Matrix{Float64} # Trial circulation
    reg::R
    get_trial_state!::G
    boundary_forces!::B
    project_circ!::C
    vort2flux!::V
end

function Solver(prob::Problem{<:StreamFcnFluid{<:StreamFcnGrid}}, state::State)
    bodies = bodygroup(prob)
    fluid = fluidof(prob)
    domain = domainof(fluid)
    @unpack nx, ny, nΓ, nq = domain

    nlevel = (nlevels ∘ discretize)(fluid)

    # TODO: Overlap memory when possible

    qs = zeros(nq, nlevel)
    Γs = zeros(nΓ, nlevel)
    Γbc = zeros(2 * (nx + 1) + 2 * (ny + 1))
    Γtmp = zeros(nΓ, nlevel)
    ψtmp = zeros(nΓ, nlevel)
    qtmp = zeros(nq, nlevel)
    Ftmp = zeros(2 * npoints(bodies))
    Γtmp1 = zeros(nΓ)
    Γtmp2 = zeros(nΓ)
    Γtmp3 = zeros(nΓ)
    qtmp1 = zeros(nq)
    qtmp2 = zeros(nq)

    C = C_linearmap(domain)

    lap_inv = LaplacianInv(domain)
    Δinv = lap_inv_linearmap(lap_inv)

    vort2flux = Vort2Flux(; domain, Δinv, ψbc=Γbc, Γtmp=Γtmp1)

    rhs_force = RhsForce(; domain, Q=qtmp1)
    nonlinear = Nonlinear(; rhs_force, C, fq=qtmp2)

    # reg must be updated if bodies move relative to the grid
    reg = Reg(domain, quantities(state).points)
    E = E_linearmap(reg)

    As, Ainvs = A_Ainv_linearmaps(prob, lap_inv)

    B_times = B_Times(;
        vort2flux, Ainv=Ainvs[1], E, C, Γtmp=Γtmp2, qtmp=qtmp2, Γ=Γtmp, ψ=ψtmp, q=qtmp
    )
    B = B_linearmap(prob, B_times)
    Binv = Binv_linearmap(prob, B)

    get_trial_state = GetTrialState(;
        prob, nonlinear, vort2flux, As, Ainvs, rhsbc=Γtmp2, rhs=Γtmp3, bc=Γbc
    )

    boundary_forces = BoundaryForcer(; prob, Binv, E, Ftmp=Ftmp, Q=qtmp1)

    project_circ = CircProjecter(; Ainv=Ainvs[1], C, E, Γtmp=Γtmp2, qtmp=qtmp1)

    return Solver(
        prob, qs, Γs, reg, get_trial_state, boundary_forces, project_circ, vort2flux
    )
end

solvertype(::Problem{<:StreamFcnFluid{<:StreamFcnGrid},<:Any,<:SchemeCNAB}) = Solver

function advance!(state::State, solver::Solver, t::Float64)
    prob = solver.prob
    fluid = fluidof(prob)
    bodies = bodygroup(prob)
    qty = quantities(state)

    # Trial flux and circulation
    @unpack qs, Γs = solver

    update_bodies!(qty.points, bodies, t)

    # Update the regularization matrix if the bodies may be moving
    eltype(bodies) <: RigidBody{GridFrame} || update!(solver.reg, qty.points)

    # Base flux from freestream and grid frame movement
    base_flux!(qty, fluid, t)

    # Computes trial circulation Γs and associated strmfcn and vel flux that don't satisfy
    # no-slip (from explicitly treated terms)
    solver.get_trial_state!(qs, Γs, state)

    # Update surface quantities to be able to trim off part of circ that doesn't satisfy no
    # slip
    @views solver.boundary_forces!(qty.Fb, qs[:, 1], qty.q0[:, 1], qty.points)

    # Compute and store integral quantities
    update_stress!(qty, prob)

    # Update circulation , vel-flux, and strmfcn on fine grid to satisfy no-slip
    solver.project_circ!(qty, Γs)

    # Interpolate values from finer grid to center region of coarse grid
    solver.vort2flux!(qty)

    state.t = t

    return nothing
end
