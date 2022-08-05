using Parameters
using StaticArrays

struct RigidBody{F<:AbstractFrame} <: AbstractBody{F}
    frame::F
    xb::Matrix{Float64} # (x, y) body points
    ds::Vector{Float64} # body segment lengths
end

npoints(body::RigidBody) = size(body.xb, 1)
