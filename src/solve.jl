"""
    StopIteration

Returned by a [`StateCallback`](@ref) to indicate that [`solve`](@ref) should terminate.
"""
struct StopIteration end

"""
    Timesteps

Specifies certain timesteps during a simulation.
"""
abstract type Timesteps end

"""
    AllTimesteps() :: Timesteps

Specifies all the timesteps during a simulation.
"""
struct AllTimesteps <: Timesteps end

"""
    TimestepIndices(i) :: Timesteps

Specifies certain timestep indices during a simulation.
"""
struct TimestepIndices{X} <: Timesteps
    i::X
end

"""
    TimestepTimes(t) :: Timesteps

Specifies certain times during a simulation.
"""
struct TimestepTimes{X} <: Timesteps
    t::X
end

"""
    StateCallback(f, timesteps::Timesteps, ret::Type=Any)

Specifies a callback `f(state)` that should be run at the given `timesteps`.

It is recommended to use [`each_timestep`](@ref), [`at_times`](@ref), or
[`at_indices`](@ref) to construct this type. The return type of `f` can be specified with
`ret`. The value of `ret` determines the type of [`StateResult`](@ref) that a callback is
mapped to.

Can be passed to [`solve`](@ref) to specify callbacks and results.
"""
struct StateCallback{T<:Timesteps,F,R}
    f::F
    timesteps::T
    ret::Type{R} # return type of f
end

StateCallback(f, timesteps) = StateCallback(f, timesteps, Any)

"""
    each_timestep(f, [ret]) :: StateCallback

Return a [`StateCallback`](@ref) that runs `f(state)` at each timestep.

`ret` specifies the return type of `f`.

See also [`at_times`](@ref), [`at_indices`](@ref).
"""
each_timestep(f, args...) = StateCallback(f, AllTimesteps(), args...)

"""
    at_times(f, t, [ret]) :: StateCallback

Return a [`StateCallback`](@ref) that runs `f(state)` at each time in `t`.

`ret` specifies the return type of `f`.

See also [`each_timestep`](@ref), [`at_indices`](@ref).
"""
at_times(f, t, args...) = StateCallback(f, TimestepTimes(t), args...)

"""
    at_indices(f, i, [ret]) :: StateCallback

Return a [`StateCallback`](@ref) that runs `f(state)` at the n'th timestep for each n in `i`.

`ret` specifies the return type of `f`.

See also [`each_timestep`](@ref), [`at_times`](@ref).
"""
at_indices(f, i, args...) = StateCallback(f, TimestepIndices(i), args...)

# Estimate how many timesteps a StateCallback will save
_timestep_count(times, cb::StateCallback) = _timestep_count(times, cb.timesteps)
_timestep_count(times::AbstractRange, ::AllTimesteps) = length(times)

function _timestep_count(::AbstractRange, t::TimestepIndices{X}) where {X}
    return Base.IteratorSize(X) == Base.HasLength() ? length(t.i) : missing
end

function _timestep_count(::AbstractRange, t::TimestepTimes{X}) where {X}
    return Base.IteratorSize(X) == Base.HasLength() ? length(t.t) : missing
end

# Return a function that runs every iteration, only calling the callback when it should
_callback_caller(_, cb::StateCallback{<:AllTimesteps}) = (_, state) -> cb.f(state)

function _callback_caller(_, cb::StateCallback{<:TimestepIndices})
    index_iter = Iterators.Stateful(cb.timesteps.i)

    return function (i, state)
        if !isempty(index_iter) && peek(index_iter) == i
            popfirst!(index_iter)
            return cb.f(state)
        end

        return nothing
    end
end

function _callback_caller(dt::Float64, cb::StateCallback{<:TimestepTimes})
    time_iter = Iterators.Stateful(cb.timesteps.t)
    half_dt = dt / 2

    return function (_, state)
        call = false
        while !isempty(time_iter) && peek(time_iter) < state.t + half_dt
            popfirst!(time_iter)
            call = true
        end

        return call ? cb.f(state) : nothing
    end
end

"""
    StateResult{T} <: AbstractVector{T}

The congregated results from a [`StateCallback`](@ref). `T` is determined by the `ret`
argument to [`StateCalllback`](@ref).

The `t` field stores the time corresponding to each data entry. `result[i]` was taken at
simulation time `result.t[i]`.
"""
struct StateResult{T} <: AbstractVector{T}
    data::Vector{T}
    t::Vector{Float64}
end

Base.size(r::StateResult) = size(r.data)
Base.getindex(r::StateResult, i) = r.data[i]
Base.IndexStyle(::Type{StateResult}) = IndexLinear()
function Base.sizehint!(r::StateResult, n)
    sizehint!(r.data, n)
    sizehint!(r.t, n)
    return r
end

function _init_result(times, cb::StateCallback)
    result = StateResult(cb.ret[], Float64[])

    n = _timestep_count(times, cb)
    !ismissing(n) && sizehint!(result, n)

    return result
end

function _result_pusher!(result::StateResult, times, cb::StateCallback)
    function pusher(state)
        u = cb.f(state)
        if !isnothing(u)
            push!(result.data, something(u))
            push!(result.t, state.t)
        end
        return nothing
    end

    return _callback_caller(times, StateCallback(pusher, cb.timesteps))
end

"""
    solve!(state, problem, (t0, tf); [save], [call]) -> results

Solve `problem` between times `t0` and `tf` by overwriting `state` each iteration.

See also [`solve`](@ref).
"""
function solve!(state::AbstractState, prob::Problem, (t0, tf)::NTuple{2}; save=(), call=())
    solver = initsolver(prob, state)

    dt = timestep(prob)
    ts = t0:dt:tf

    callers = map(cb -> _callback_caller(dt, cb), call)
    results = map(cb -> _init_result(ts, cb), save)
    pushers = map((cb, r) -> _result_pusher!(r, ts, cb), save, results)

    for (i, t) in enumerate(ts)
        advance!(state, solver, t)

        # Push results
        for p in pushers
            p(i, state)
        end

        # Break if any callback returns StopIteration(), but call the remaining callbacks
        stop = false
        for f in callers
            stop |= f(i, state) == StopIteration()
        end
        stop && break
    end

    return results
end

"""
    solve(problem, (t0, tf); [save], [call]) -> results

Solve `problem` between times `t0` and `tf`, optionally saving or executing callbacks.

# Arguments
- `problem::Problem`: The problem specification.
- `(t0, tf)`: Initial and final times.
- `save`: Collection of [`StateCallback`](@ref) that maps to `results` (default is empty).
- `call`: Collection of [`StateCallback`](@ref) to be executed during the simulation.

# Returns
The result of mapping each callback in `save` to a [`StateResult`](@ref) using `map`.
"""
solve(prob, tlims; kw...) = solve!(initstate(prob, tlims[1]), prob, tlims; kw...)
