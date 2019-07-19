using HomotopyContinuation
using LinearAlgebra
using DynamicPolynomials

include("abbadingo.jl")
include("util.jl")
k = 8

train = read_dct("dcts/dfa_8_try_1.dct")

@polyvar δ[1:2, 1:k, 1:k]
@polyvar z[1:k]

q0 = zeros(k)
q0[1] = 1

equations = DynamicPolynomials.Polynomial{true,Float64}[]
for (seq, accept) in train
    push!(equations, wfa_eval(q0, z, [δ[1, :, :], δ[2, :, :]], (1,0,1,1)) - accept)
end

# lol
