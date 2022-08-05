# TODO: Throw useful error when body is outside innermost grid
# TODO: In State, store bodypoints as views to one big array to avoid copying here

struct Reg{D<:StreamFcnGrid}
    domain::D
    body_idx::Matrix{Int}
    supp_idx::UnitRange{Int}
    weight::Array{Float64,4}
end

struct RegT{F}
    reg::Reg{F}
end

function E_linearmap(reg::Reg)
    regT = RegT(reg)

    nf = size(reg.weight, 1)
    nq = reg.domain.nq

    return LinearMap(regT, reg, nf, nq)
end

function Reg(domain::StreamFcnGrid, points::BodyGroupPoints)
    nf = length(points.r)
    body_idx = zeros(Int, size(points.r))

    # TODO: Add external option for supp
    supp = 6
    supp_idx = -supp:supp
    weight = zeros(nf, 2, 2 * supp + 1, 2 * supp + 1)

    reg = Reg(domain, body_idx, supp_idx, weight)
    return update!(reg, points)
end

function update!(reg::Reg, points::BodyGroupPoints)
    @unpack domain, body_idx, supp_idx, weight = reg

    xb = points.r
    nb = size(xb, 1)

    basegrid = (baselevel ∘ discretize)(domain)
    h = gridstep(basegrid)
    x0, y0 = minimum.(xycoords(basegrid))

    # Nearest indices of body relative to grid
    @views @. body_idx[:, 1] = floor(Int, (xb[:, 1] - x0) / h)
    @views @. body_idx[:, 2] = floor(Int, (xb[:, 2] - y0) / h)

    # get regularized weight near IB points (u-vel points)
    for k in 1:nb
        x = @. h * (body_idx[k, 1] - 1 + supp_idx) + x0
        y = permutedims(@. h * (body_idx[k, 2] - 1 + supp_idx) + y0)

        @. weight[k, 1, :, :] = δh(x, xb[k, 1], h) * δh(y + h / 2, xb[k, 2], h)
        @. weight[k, 2, :, :] = δh(x + h / 2, xb[k, 1], h) * δh(y, xb[k, 2], h)
    end

    return reg
end

function (reg::Reg)(q_flat, fb_flat)
    # Matrix E'
    @unpack domain, body_idx, supp_idx, weight = reg
    nb = size(body_idx, 1)

    q_flat .= 0
    fb = reshape(fb_flat, nb, 2)
    qx, qy = split_flux(q_flat, domain)

    for k in 1:nb
        i = body_idx[k, 1] .+ supp_idx
        j = body_idx[k, 2] .+ supp_idx
        @views @. qx[i, j] += weight[k, 1, :, :] * fb[k, 1]
        @views @. qy[i, j] += weight[k, 2, :, :] * fb[k, 2]

        # Should be safe to remove this now
        # TODO: Throw proper exception or remove
        if !isfinite(sum(x -> x^2, qx[i, j]))
            error("infinite flux")
        end
    end

    return nothing
end

function (regT::RegT)(fb_flat, q_flat)
    # Matrix E
    @unpack domain, body_idx, supp_idx, weight = regT.reg
    nb = size(body_idx, 1)

    fb_flat .= 0
    fb = reshape(fb_flat, nb, 2)
    qx, qy = split_flux(q_flat, domain)

    for k in 1:nb
        i = body_idx[k, 1] .+ supp_idx
        j = body_idx[k, 2] .+ supp_idx
        fb[k, 1] += @views sumprod(qx[i, j], weight[k, 1, :, :])
        fb[k, 2] += @views sumprod(qy[i, j], weight[k, 2, :, :])
    end

    return nothing
end

sumprod(xs, ys) = sum(x * y for (x, y) in zip(xs, ys))

"""
    δh( rf, rb , dr )

Discrete delta function used to relate flow to structure quantities
"""
function δh(rf, rb, dr)
    # take points on the flow domain (r) that are within the support
    # (supp) of the IB points (rb), and evaluate delta( abs(r - rb) )

    # Currently uses the Yang3 smooth delta function (see Yang et al, JCP, 2009),
    # which has a support of 6*h (3*h on each side)

    # Note that this gives slightly different answers than Fortran at around 1e-4,
    # apparently due to slight differences in the floating point arithmetic.  As far as I
    # can tell, this is what sets the bound on agreement between the two implementations.
    # It's possible this might be improved with arbitrary precision arithmetic (i.e.
    # BigFloats), but at least it doesn't seem to be a bug.

    # Note: the result is delta * h
    r = abs(rf - rb)
    r1 = r / dr
    r2 = r1 * r1
    r3 = r2 * r1
    r4 = r3 * r1

    return if (r1 <= 1.0)
        #println("r < 1")
        a5 = asin((1.0 / 2.0) * sqrt(3.0) * (2.0 * r1 - 1.0))
        a8 = sqrt(1.0 - 12.0 * r2 + 12.0 * r1)

        4.166666667e-2 * r4 +
        (-0.1388888889 + 3.472222222e-2 * a8) * r3 +
        (-7.121664902e-2 - 5.208333333e-2 * a8 + 0.2405626122 * a5) * r2 +
        (-0.2405626122 * a5 - 0.3792313933 + 0.1012731481 * a8) * r1 +
        8.0187537413e-2 * a5 - 4.195601852e-2 * a8 + 0.6485698427

    elseif (r1 <= 2.0)
        #println("r < 2")
        a6 = asin((1.0 / 2.0) * sqrt(3.0) * (-3.0 + 2.0 * r1))
        a9 = sqrt(-23.0 + 36.0 * r1 - 12.0 * r2)

        -6.250000000e-2 * r4 +
        (0.4861111111 - 1.736111111e-2 * a9) .* r3 +
        (-1.143175026 + 7.812500000e-2 * a9 - 0.1202813061 * a6) * r2 +
        (0.8751991178 + 0.3608439183 * a6 - 0.1548032407 * a9) * r1 - 0.2806563809 * a6 +
        8.22848104e-3 +
        0.1150173611 * a9

    elseif (r1 <= 3.0)
        #println("r < 3")
        a1 = asin((1.0 / 2.0 * (2.0 * r1 - 5.0)) * sqrt(3.0))
        a7 = sqrt(-71.0 - 12.0 * r2 + 60.0 * r1)

        2.083333333e-2 * r4 +
        (3.472222222e-3 * a7 - 0.2638888889) * r3 +
        (1.214391675 - 2.604166667e-2 * a7 + 2.405626122e-2 * a1) * r2 +
        (-0.1202813061 * a1 - 2.449273192 + 7.262731481e-2 * a7) * r1 +
        0.1523563211 * a1 +
        1.843201677 - 7.306134259e-2 * a7
    else
        0.0
    end
end
