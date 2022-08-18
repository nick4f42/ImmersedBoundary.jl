module StreamFcn

using LinearAlgebra
using LinearMaps
using Parameters
using FFTW
using EllipsisNotation
using StaticArrays
using FunctionWrappers: FunctionWrapper as FuncWrap

using ..ImmersedBoundary
using ..ImmersedBoundary.Dynamics
using ..ImmersedBoundary.Fluids
using ..ImmersedBoundary.Bodies
import ..ImmersedBoundary: domainof, frameof, discretize

export StreamFcnFluid, StreamFcnState, StreamFcnQuantities, StreamFcnGrid
export quantities, update_stress!
export flatten_circ, unflatten_circ, split_flux
export coarsify!, get_bc!, apply_bc!
export curl!, C_linearmap, lap_inv_linearmap, LaplacianInv, E_linearmap, Reg, update!
export RhsForce, Vort2Flux, base_flux!, avg_flux!, direct_product!

include("fluids.jl")
include("states.jl")
include("fluid-ops.jl")
include("structure-ops.jl")

include("cnab/cnab.jl")

end # module
