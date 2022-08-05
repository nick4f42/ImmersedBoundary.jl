module Dynamics

using FunctionWrappers: FunctionWrapper as FuncWrap
using StaticArrays
using Parameters
using Base: AbstractVecOrTuple

export AbstractFrame, RootFrame, GridFrame, LabFrame
export RelativeFrame, RelativeFrameInstant
# export coords, position, velocity

abstract type AbstractFrame end
abstract type RootFrame <: AbstractFrame end

struct GridFrame <: RootFrame end
struct LabFrame <: RootFrame end

const Scalar = Float64
const Coords = SVector{2,Scalar}
const CoordsFunc = FuncWrap{Coords,Tuple{Scalar}}
const ScalarFunc = FuncWrap{Scalar,Tuple{Scalar}}

struct RelativeFrame{F<:RootFrame} <: AbstractFrame
    base::F
    r::CoordsFunc # position
    v::CoordsFunc # velocity
    θ::ScalarFunc # angular position
    Ω::ScalarFunc # angular velocity
end

struct RelativeFrameInstant{F<:RootFrame}
    base::F
    r::Coords
    v::Coords
    θ::Scalar
    Ω::Scalar
    cθ::Scalar # cos(θ)
    sθ::Scalar # sin(θ)
end

function RelativeFrameInstant(base, r, v, θ, Ω)
    return RelativeFrameInstant(base, r, v, θ, Ω, cos(θ), sin(θ))
end

function (frame::RelativeFrame)(t::Real)
    base = frame.base
    return RelativeFrameInstant(base, base.r(t), base.v(t), base.θ(t), base.Ω(t))
end

function Base.:/(a::F, b::F) where {F<:RelativeFrameInstant}
    r = a.r - b.r
    v = a.v - b.v
    θ = a.θ - b.θ
    Ω = a.Ω - b.Ω
    return RelativeFrameInstant(a.base, r, v, θ, Ω)
end

struct Vec2D{F<:AbstractFrame}
    frame::F
    x::Coords
end

Vec2D(::F, v::Vec2D{F}) where {F<:RootFrame} = v

function Vec2D(frame::F, v::Vec2D{F}) where {F<:RelativeFrameInstant}
    return if frame == v.frame
        v
    else
        rel = v.frame / frame
        cθ, sθ = rel.cθ, rel.sθ
        Vec2D(frame, SA[cθ -sθ; sθ cθ] * v.x)
    end
end

function Vec2D(frame::F, v::Vec2D{RelativeFrameInstant{F}}) where {F}
    cθ, sθ = v.frame.cθ, v.frame.sθ
    return Vec2D(frame, SA[cθ -sθ; sθ cθ] * v.x)
end

function Vec2D(frame::RelativeFrameInstant{F}, v::Vec2D{F}) where {F}
    cθ, sθ = frame.cθ, frame.sθ
    return Vec2D(frame, SA[cθ sθ; -sθ cθ] * v.x)
end

coords(v::Vec2D) = v.x

coords((a, b)::Pair, v::Coords) = coords(Vec2D(b, Vec2D(a, v)))

# TODO: Tools for converting position/velocity between frames

cross2d(a::Scalar, b::Coords) = a * rot90(b)
rot90((x, y)::Coords) = Coords(-y, x)

end # module
