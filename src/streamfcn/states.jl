abstract type StreamFcnState <: AbstractState end

struct BodyStress
    fb::Matrix{Float64} # Surface stresses
    CD::Float64 # Drag coeff
    CL::Float64 # Lift coeff
end

struct StreamFcnQuantities <: AbstractQuantities
    q::Matrix{Float64} # Flux
    q0::Matrix{Float64} # Base flux
    Γ::Matrix{Float64} # Circulation
    ψ::Matrix{Float64} # Streamfunction
    Fb::Vector{Float64} # Flattened (body forces * dt)
    stress::Vector{BodyStress}
    points::BodyGroupPoints
end

function StreamFcnQuantities(prob::Problem{<:StreamFcnFluid{<:StreamFcnGrid}}, t::Real)
    fluid = fluidof(prob)
    bodies = bodygroup(prob)
    @unpack nq, nΓ = domainof(fluid)

    nlevel = (nlevels ∘ discretize)(fluid)

    nb = map(npoints, bodies)

    q = zeros(nq, nlevel)  # Flux
    q0 = zeros(nq, nlevel) # Background flux
    Γ = zeros(nΓ, nlevel)  # Circulation
    ψ = zeros(nΓ, nlevel)  # Streamfunction
    Fb = zeros(2 * npoints(bodies))

    stress = [BodyStress(zeros(n, 2), 0.0, 0.0) for n in nb]
    points = BodyGroupPoints(bodies)

    qty = StreamFcnQuantities(q, q0, Γ, ψ, Fb, stress, points)
    base_flux!(qty, fluid, t)

    return qty
end

function setnoise!(qty::StreamFcnQuantities, noise::Float64)
    for i in eachindex(qty.Γ)
        qty.Γ[i] = noise * randn()
    end
    for i in eachindex(qty.q)
        qty.q[i] = noise * randn()
    end
    return qty
end

function update_stress!(qty::StreamFcnQuantities, prob::Problem{<:StreamFcnFluid})
    # Updates the stress field to reflect Fb
    bodies = bodygroup(prob)
    h = (gridstep ∘ baselevel ∘ discretize ∘ fluidof)(prob)
    dt = timestep(prob)
    Fb = reshape(qty.Fb, :, 2)

    # Store surface stress and integrated forces
    i = 1
    for (j, body) in zip(eachindex(qty.stress), bodies)
        n = npoints(body)
        ds = body.ds

        F = @view Fb[i:(i+n-1), :]

        # Surface stress * ds
        fb = qty.stress[j].fb
        @. fb = F * h / dt

        # Integrated forces
        CD = 2 * sum(@view fb[:, 1])
        CL = 2 * sum(@view fb[:, 2])

        # Fix surface stress
        fb ./= ds

        qty.stress[j] = BodyStress(fb, CD, CL)
        i += n
    end

    return nothing
end

"Define the length of a state as the size of the circulation vector Γ"
Base.length(v::StreamFcnQuantities) = size(v.Γ, 1)

"Copy all fields of State v to w"
function Base.copy!(dst::StreamFcnQuantities, src::StreamFcnQuantities)
    copyto!(dst.q, src.q)
    copyto!(dst.q0, src.q0)
    copyto!(dst.Γ, src.Γ)
    copyto!(dst.ψ, src.ψ)
    copyto!(dst.Fb, src.Fb)
    copyto!(dst.stress, src.stress)
    copy!(dst.points, src.points)

    return dst
end

function Base.similar(v::StreamFcnQuantities)
    q = similar(v.q)
    q0 = similar(v.q0)
    Γ = similar(v.Γ)
    ψ = similar(v.ψ)
    Fb = similar(v.Fb)
    stress = similar(v.stress)
    points = similar(v.points)

    return StreamFcnQuantities(q, q0, Γ, ψ, Fb, stress, points)
end

"Out of place scalar multiplication; multiply vector v with scalar α and store the result in w"
function LinearAlgebra.mul!(w::StreamFcnQuantities, v::StreamFcnQuantities, α::Real)
    @. w.Γ = v.Γ * α
    @. w.q = v.q * α
    @. w.ψ = v.ψ * α
    return w
end

"In-place scalar multiplication of v with α; in particular with α = false, v is the corresponding zero vector"
function LinearAlgebra.rmul!(v::StreamFcnQuantities, α::Real)
    v.Γ .*= α
    v.q .*= α
    v.ψ .*= α
    return v
end

function Base.:*(v::StreamFcnQuantities, α::Real)
    w = deepcopy(v)
    rmul!(w, α)
    return w
end

Base.:*(α::Real, v::StreamFcnQuantities) = v * α

Base.:*(α::Real, v::StreamFcnState) = v * α
function Base.:*(v::StreamFcnState, α::Real)
    w = similar(v)
    return mul!(w, v, α)
end

function LinearAlgebra.axpby!(
    α::Real, v::StreamFcnQuantities, β::Real, w::StreamFcnQuantities
)
    axpby!(α, v.q, β, w.q)
    axpby!(α, v.Γ, β, w.Γ)
    axpby!(α, v.ψ, β, w.ψ)
    return w
end

function LinearAlgebra.axpy!(α::Real, v::StreamFcnQuantities, w::StreamFcnQuantities)
    axpy!(α, v.q, w.q)
    axpy!(α, v.Γ, w.Γ)
    axpy!(α, v.ψ, w.ψ)
    return w
end

LinearAlgebra.dot(v::StreamFcnQuantities, w::StreamFcnQuantities) = dot(v.q, w.q)
function LinearAlgebra.dot(v::S, w::S) where {S<:StreamFcnState}
    return dot(quantities(v), quantities(w))
end

LinearAlgebra.norm(v::StreamFcnQuantities) = norm(v.q)
LinearAlgebra.norm(v::StreamFcnState) = norm(quantities(v))
