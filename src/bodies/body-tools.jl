function circle((x, y)::NTuple{2}, r, h; frame=GridFrame())
    # Choose n points such that ds≈2h
    n = floor(Int, π * r / h)
    t = 2π / n * (0:(n-1))

    xb = @. [(x + r * cos(t)) (y + r * sin(t))]
    ds = hypot(xb[2, 1] - xb[1, 1], xb[2, 2] - xb[1, 2])

    return RigidBody(frame, xb, fill(ds, n))
end
