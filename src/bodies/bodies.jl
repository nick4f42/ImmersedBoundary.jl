module Bodies

using ..ImmersedBoundary.Dynamics

export AbstractBody, BodyGroup, npoints, RigidBody
export BodyPoints, BodyGroupPoints, update_bodies!
export circle

abstract type AbstractBody{F} end

struct BodyGroup{B} <: AbstractVector{B}
    b::Vector{B}
    npoints::Int
    function BodyGroup(bodies::AbstractVector{B}) where {B<:AbstractBody}
        n = sum(npoints, bodies)
        return new{B}(bodies, n)
    end
end

BodyGroup(body::AbstractBody...) = BodyGroup([body...])
BodyGroup(bodies) = BodyGroup(collect(bodies))

Base.size(bodies::BodyGroup) = size(bodies.b)
Base.getindex(bodies::BodyGroup, i) = bodies.b[i]
Base.IndexStyle(::BodyGroup) = IndexCartesian()

"""
    npoints(body::AbstractBody)
    npoints(bodies::BodyGroup)

Number of discrete points on a body or bodies.
"""
function npoints end

npoints(bodies::BodyGroup) = bodies.npoints

include("rigid-bodies.jl")
include("body-points.jl")
include("body-tools.jl")

end # module
