using GLPK
using Cbc
using JuMP
using LowRankApprox
using LinearAlgebra
using Clustering
using Statistics
using IterTools
using SimpleGraphs
using SimpleGraphAlgorithms
d = 7
s = 2

P = vcat([collect.(collect(Iterators.product(repeat([collect(0:s-1)], l)...))[:]) for l in 0:ceil(Int, d/2 + 2)]...)

function array_mod(a, base, mod)
    result = 0
    for i in 1:length(a)
        result *= base
        result += a[i]
        result %= mod
    end
    return result
end


function dfa(x)
    return Int(array_mod(x, 2, d) == 0) + Int(array_mod(x, 2, d) == 4)
end

# function dfa(x)
#    if length(x) < 3
#        return 0
#    end

#    return Int(x[end-2] == 1)
# end

# function dfa(x)
#    switches = 0
#    for i in 2:length(x)
#        if x[i] != x[i - 1]
#            switches += 1
#        end
#    end
#    return switches % 3 == 0
# end

Hf = hcat([[dfa(vcat(p, s)) for s in P] for p in P]...)'
Hf0 = hcat([[dfa(vcat(p, [0], s)) for s in P] for p in P]...)'
Hf1 = hcat([[dfa(vcat(p, [1], s)) for s in P] for p in P]...)'


mask = randn(size(Hf)) .> 1.5

states = [Hf[1,:]]
masks = [mask[1,:]]

function rows(A)
    n, m = size(A)
    return [A[i,:] for i in 1:n]
end

function cols(A)
    n, m = size(A)
    return [A[:,i] for i in 1:m]
end

n, m = size(Hf)

G = IntGraph(n)
for (i, j) in subsets(1:n, 2)
    if any((mask[i,:] .& mask[j,:]) .* (Hf[i,:] .!= Hf[j,:]))
        add!(G,i,j)
    end
end

use_optimizer(Cbc, Dict(:logLevel=>1, :threads=>Sys.CPU_THREADS))

function min_color(G::SimpleGraph, h=maximum(values(greedy_color(G))))
    VV = vlist(G)
    EE = elist(G)
    n = NV(G)
    m = NE(G)

    err_msg = "This graph is not $(h) colorable"

    MOD = Model(with_optimizer(Cbc.Optimizer; logLevel=1, threads=Sys.CPU_THREADS))
    # MOD = Model(with_optimizer(GLPK.Optimizer, msg_lev=GLPK.MSG_ON))

    @variable(MOD, x[VV,1:h], Bin)
    @variable(MOD, w[1:h], Bin)

    for v in VV
        @constraint(MOD, sum(x[v,:]) == 1)
    end

    for (u, v) in EE
        for i=1:h
            @constraint(MOD, x[u,i] + x[v,i] <= w[i])
        end
    end

    for i in 1:h
        @constraint(MOD, w[i] <= sum(x[:,i]))
    end

    for i in 2:h
        @constraint(MOD, w[i] <= w[i-1])
    end

    @objective(MOD, MOI.MIN_SENSE, sum(w))

    optimize!(MOD)
    status = Int(termination_status(MOD))

    if status != 1
        error(err_msg)
    end

    X = value.(x)

    result = Dict{vertex_type(G),Int}()

    for v in VV
        for c = 1:h
            if X[v,c] > 0
                result[v] = c
            end
        end
    end

    return result
end

function min_color2(G::SimpleGraph, h=maximum(values(greedy_color(G))))
    VV = vlist(G)
    EE = elist(G)
    n = NV(G)
    m = NE(G)

    err_msg = "This graph is not $(h) colorable"

    MOD = Model(with_optimizer(Cbc.Optimizer; logLevel=1, threads=Sys.CPU_THREADS))

    @variable(MOD, y[1:h,VV], Bin)
    @variable(MOD, z[VV,1:h], Bin)

    q=1

    for v in VV
        @constraint(MOD, z[v,1] == 0)
        @constraint(MOD, y[h,v] == 0)

        for i in 1:h-1
            @constraint(MOD, y[i,v] - y[i+1,v] >= 0)
            @constraint(MOD, y[i,v] + z[v,i+1] == 1)
            @constraint(MOD, y[i,q] - y[i,v] >= 0)
        end
    end

    for (u, v) in EE
        for i in 1:h
            @constraint(MOD, y[i,u] + z[u,i] + y[i,v] + z[v,i] >= 1)
        end
    end

    @objective(MOD, MOI.MIN_SENSE, 1+sum(y[:,q]))

    MOD
end

# @show min_color(G)
