struct BodyPoints
    r::Matrix{Float64} # (x, y) body points
    v::Matrix{Float64} # Point velocities
end

struct BodyGroupPoints
    pts::Vector{BodyPoints}
    r::Matrix{Float64} # stacked (x, y) body points
    v::Matrix{Float64} # stacked point velocities
end

function BodyPoints(body::RigidBody)
    xb = body.xb
    return BodyPoints(copy(xb), zeros(size(xb)))
end

function BodyGroupPoints(bodies::BodyGroup)
    pts = map(BodyPoints, bodies)

    nb = npoints(bodies)
    r = zeros(nb, 2)
    v = zeros(nb, 2)
    vcat!(r, (p.r for p in pts))

    return BodyGroupPoints(pts, r, v)
end

function Base.copy!(dst::BodyPoints, src::BodyPoints)
    copyto!(dst.r, src.r)
    copyto!(dst.v, src.v)
    return dst
end

function Base.copy!(dst::BodyGroupPoints, src::BodyGroupPoints)
    copyto!(dst.r, src.r)
    copyto!(dst.v, src.v)
    for (dst_pts, src_pts) in zip(dst.pts, src.pts)
        copyto!(dst_pts, src_pts)
    end

    return dst
end

Base.similar(p::BodyPoints) = BodyPoints(similar(p.r), similar(p.v))
Base.similar(p::BodyGroupPoints) = BodyGroupPoints(similar.((p.pts, p.r, p.v))...)

update_bodies!(::BodyPoints, ::RigidBody{GridFrame}, _) = nothing

function update_bodies!(
    pts::BodyPoints, base::RigidBody{RelativeFrame{GridFrame}}, t::Float64
)
    r0, v0, θ, ω = map(f -> getfield(base.frame, f)(t), (:r, :v, :θ, :ω))
    c = cos(θ)
    s = sin(θ)

    Rx = SA[c -s; s c]
    Rv = ω * SA[-s -c; c -s]

    for (r, v, rb) in zip(eachrow(pts.r), eachrow(pts.v), base.xb)
        r .= r0 + Rx * rb
        v .= v0 + Rv * rb
    end

    return nothing
end

function update_bodies!(b::BodyGroupPoints, base::BodyGroup{B}, t::Float64) where {B}
    for (p, body) in zip(b.pts, base)
        update_bodies!(p, body, t)
    end

    vcat!(b.r, (p.r for p in b.pts))
    vcat!(b.v, (p.r for p in b.pts))

    return nothing
end

"""
    vcat!(y, xs)

Concatenate rows of each matrix in `xs` into matrix `y`.
"""
function vcat!(y::AbstractMatrix, xs)
    i = firstindex(y, 1)
    for x in xs
        n = size(x, 1)
        y[i:(i+n-1), :] = x
        i += n
    end
    return y
end
