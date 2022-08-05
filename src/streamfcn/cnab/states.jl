mutable struct State <: StreamFcnState
    qty::StreamFcnQuantities
    nonlin::Vector{Matrix{Float64}} # Memory of nonlinear terms
    cfl::Float64
    t::Float64
end

function State(prob::Problem{<:StreamFcnFluid}, t::Real)
    nΓ = (domainof ∘ fluidof)(prob).nΓ
    nlevel = (nlevels ∘ discretize ∘ fluidof)(prob)
    nstep = length(schemeof(prob).β)

    qty = StreamFcnQuantities(prob, t)
    nonlin = [zeros(nΓ, nlevel) for _ in 1:nstep]
    return State(qty, nonlin, 0.0, t)
end

statetype(::Problem{<:StreamFcnFluid{<:StreamFcnGrid},<:Any,<:SchemeCNAB}) = State
quantities(s::State) = s.qty

Base.length(s::State) = length(quantities(s))

function Base.similar(state::State)
    qty = similar(state.qty)
    nonlin = map(similar, state.nonlin)
    return State(qty, nonlin, 0, 0)
end

"Out of place scalar multiplication; multiply vector v with scalar α and store the result in w"
function LinearAlgebra.mul!(w::State, v::State, α::Real)
    mul!(quantities(w), quantities(v), α)
    for (wn, vn) in zip(w.nonlin, v.nonlin)
        @. wn = vn * α
    end
    return w
end

"In-place scalar multiplication of v with α; in particular with α = false, v is the corresponding zero vector"
function LinearAlgebra.rmul!(v::State, α::Real)
    rmul!(quantities(v), α)
    for n in v.nonlin
        n .*= α
    end
    return v
end

"store in w the result of α*v + β*w"
function LinearAlgebra.axpby!(α::Real, v::State, β::Real, w::State)
    axpby!(α, quantities(v), β, quantities(w))
    for (vn, wn) in zip(v.nonlin, w.nonlin)
        axpby!(α, vn, β, wn)
    end
    return w
end

function LinearAlgebra.axpy!(α::Real, v::State, w::State)
    axpy!(α, quantities(v), quantities(w))
    for (vn, wn) in zip(v.nonlin, w.nonlin)
        axpy!(α, vn, wn)
    end
    return w
end
