abstract type AbstractScheme end

"""
    SchemeRKC(dt) :: AbstractScheme

Runge-Kutta-Chebyshev scheme.
"""
struct SchemeRKC <: AbstractScheme
    dt::Float64
end

timestep(scheme::SchemeRKC) = scheme.dt

"""
    SchemeCNAB(Val(N), dt) :: AbstractScheme

N-step Crank-Nicolson/Adams-Bashforth scheme.
"""
struct SchemeCNAB <: AbstractScheme
    β::Vector{Float64} # adams bashforth coefficients
    dt::Float64
    SchemeCNAB(::Val{2}, dt) = new([1.5, -0.5], dt)
end

# Default to 2-step
SchemeCNAB(dt) = SchemeCNAB(Val(2), dt)

timestep(scheme::SchemeCNAB) = scheme.dt
