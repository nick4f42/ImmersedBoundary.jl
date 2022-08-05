"""
    CNAB

Implements the Crank-Nicolson/Adams-Bashforth scheme for the streamfcn-vorticity
formulation.
"""
module CNAB

using LinearAlgebra
using Parameters
using LinearMaps
using IterativeSolvers

using ...ImmersedBoundary
using ...ImmersedBoundary.Dynamics
using ...ImmersedBoundary.Bodies
using ...ImmersedBoundary.Fluids
using ..StreamFcn

import ...ImmersedBoundary: advance!, quantities, statetype, solvertype

include("states.jl")
include("ops.jl")
include("timestepping.jl")

end # module
