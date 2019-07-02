using JuMP
using ProxSDP
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



model = Model(with_optimizer(ProxSDP.Optimizer, log_verbose=true, full_eig_decomp=true))

# M = Hf

M = vcat(Hf, Hf0, Hf1)
mask = randn(size(M)) .> 1.5

n, m = size(M)

@variable(model, W1[1:n, 1:n])
@variable(model, W2[1:m, 1:m])
@variable(model, X[1:n, 1:m])

@constraint(model, [W1 X; X' W2] in PSDCone())
@constraint(model, mask .* (X - M) .== 0)
@constraint(model, 1 .>= X .>= 0)

@objective(model, MOI.MIN_SENSE, tr(W1) + tr(W2))

optimize!(model)
