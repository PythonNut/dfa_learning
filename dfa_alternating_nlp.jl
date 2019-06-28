using JuMP
# using GLPK
using Ipopt
using LowRankApprox
using LinearAlgebra
using Clustering
using Statistics

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


mask = randn(size(Hf)) .> 1

X = randn(127, 7)

for i in 1:100
    global X
    model = Model(with_optimizer(Ipopt.Optimizer))
    @variable(model, Y[1:7, 1:127])
    @objective(model, MOI.MIN_SENSE, sum(mask .* (Hf - X * Y) .^2))

    optimize!(model)

    Y = value.(Y)

    model = Model(with_optimizer(Ipopt.Optimizer))
    @variable(model, X[1:127, 1:7])
    @objective(model, MOI.MIN_SENSE, sum(mask .* (Hf - X * Y) .^2))

    optimize!(model)

    X = value.(X)
end

