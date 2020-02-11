# NeuralThermalODE

This package is designed to ease my PhD work by placing in a module all the
functionality I need for extracting thermal properties from experimental test
data. It is not recommended for other work since generality is close to
nonexistent.

Currently the default implementation solves the heat equation using the ADAM
optimizer and the adjoint method to compute unknown thermal boundary
conditions.

This module is build upon DifferentialEquations.jl, Flux.jl and DiffEqFlux.jl
