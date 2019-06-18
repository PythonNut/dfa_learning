using LowRankApprox
using LinearAlgebra
using Clustering
using Statistics

d = 7
s = 2

P = vcat([collect.(collect(Iterators.product(repeat([collect(0:s-1)], l)...))[:]) for l in 0:ceil(Int, d/2 + 4)]...)

function array_mod(a, base, mod)
   result = 0
    for i in 1:length(a)
        result *= base
        result += a[i]
        result %= mod
    end
    return result
end


# function dfa(x)
#     return Int(array_mod(x, 2, d) == 0) + Int(array_mod(x, 2, d) == 4)
# end

#function dfa(x)
#    if length(x) < 3
#        return 0
#    end
#
#    return Int(x[end-2] == 1)
#end

function dfa(x)
   switches = 0
   for i in 2:length(x)
       if x[i] != x[i - 1]
           switches += 1
       end
   end
   return switches % 3 == 0
end

Hf = hcat([[dfa(vcat(p, s)) for s in P] for p in P]...)'
Hf0 = hcat([[dfa(vcat(p, [0], s)) for s in P] for p in P]...)'
Hf1 = hcat([[dfa(vcat(p, [1], s)) for s in P] for p in P]...)'

U, S, V = psvd(float.(Hf), rank=d)

pinvHfV = pinv(Hf * V)

A0 = pinvHfV * Hf0 * V
A1 = pinvHfV * Hf1 * V

A = [A0, A1]

q = dfa.(P)
a1 = (q' * V)'
a∞ = pinvHfV * q

function int2bitarray(i)
    return [parse(Int, c) for c in string(i, base=2)]
end

function wfa_eval(a1, a∞, A, s)
    return reduce(*, [a1', (A[c + 1] for c in s)..., a∞])
end

@show [wfa_eval(a1, a∞, A, int2bitarray(d*99 + k)) for k in 0:d-1]

X = vcat([reduce(*, [a1', (A[c + 1] for c in p)...]) for p in P]...)

# Estimate k
ϵ = 1e-8
clusters_approx = []
for i in 1:size(X)[1]
    if isempty(clusters_approx) || minimum([norm(X[i,:] - c) for c in clusters_approx]) > ϵ
        push!(clusters_approx, X[i,:])
    end
end

k = length(clusters_approx)
clusters = kmeans(collect(X'), k)
cluster_size = maximum([maximum(std(X[clusters.assignments .== i,:], dims=1, corrected=false)) for i in 1:k])
cluster_separation = minimum(hcat([[norm(clusters.centers[:,i] - clusters.centers[:,j]) for i in 1:k] for j in 1:k]...) + Inf*I)

@assert cluster_separation > 100*cluster_size
q1 = Int.(mapslices(norm, a1.-clusters.centers, dims=1)' .< ϵ)[:]
q∞ = round.(Int, (a∞' * clusters.centers)')
δ = [
    Int.(hcat([[norm(A[s]' * clusters.centers[:,i] - clusters.centers[:,j]) for i in 1:k] for j in 1:k]...) .< cluster_separation/2)
    for s in 1:length(A)
]

@show [wfa_eval(q1, q∞, δ, int2bitarray(d*99 + k)) for k in 0:d-1]
