module ImmersedBoundary

using Reexport

export SchemeRKC, SchemeCNAB
export AbstractState, AbstractQuantities, AbstractSolver, Problem
export fluidof, bodygroup, schemeof, timestep, quantities, discretize, problemof
export advance!, initstate, initsolver

function discretize end
function advance! end
function statetype end
function solvertype end
function quantities end
function problemof end

include("dynamics.jl")
include("schemes.jl")
include("fluids/fluids.jl")
include("bodies/bodies.jl")

@reexport begin
    using .Dynamics: AbstractFrame, GridFrame, LabFrame, RelativeFrame
    using .Fluids: AbstractFluid, AbstractFluidDomain, FluidDiscretization
    using .Fluids: FluidDomain, UniformGrid, MultiLevel, MultiUniformGrid
    using .Fluids: frameof, domainof, gridstep, xycoords, baselevel, subdomain
    using .Bodies: AbstractBody, BodyGroup, RigidBody, BodyPoints, BodyGroupPoints
end

abstract type AbstractState end
abstract type AbstractQuantities end
abstract type AbstractSolver end

struct Problem{F<:AbstractFluid,B<:BodyGroup,S<:AbstractScheme}
    fluid::F
    bodies::B
    scheme::S
end

include("solve.jl")
include("streamfcn/streamfcn.jl")

@reexport using .StreamFcn: StreamFcnFluid
export StopIteration, each_timestep, at_times, at_indices, solve, solve!

fluidof(prob::Problem) = prob.fluid
bodygroup(prob::Problem) = prob.bodies
schemeof(prob::Problem) = prob.scheme
timestep(prob::Problem) = (timestep ∘ schemeof)(prob)

initstate(prob::Problem, t::Real=0.0) = statetype(prob)(prob, t)
initsolver(prob::Problem, state::AbstractState) = solvertype(prob)(prob, state)

end # module
