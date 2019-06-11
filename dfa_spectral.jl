using LowRankApprox
using LinearAlgebra

P = collect.(collect(Iterators.product(repeat([[0, 1]], 7)...))[:])

Hf = hcat([[Int(parse(Int, join(cat(p, s, dims=1), ""), base=2) % 7 ==0) for p in P] for s in P]...)'
Hf0 = hcat([[Int(parse(Int, join(cat(p, [0], s, dims=1), ""), base=2) % 7 ==0) for p in P] for s in P]...)'
Hf1 = hcat([[Int(parse(Int, join(cat(p, [1], s, dims=1), ""), base=2) % 7 ==0) for p in P] for s in P]...)'

U, S, V = psvd(float.(Hf), rank=7)

A0 = pinv(Hf * V) * Hf0 * V
A1 = pinv(Hf * V) * Hf1 * V

A = [A0, A1]

q = [Int(parse(Int, join(p, ""), base=2) % 7 ==0) for p in P]
a1 = (q' * V)'
a∞ = pinv(Hf * V) * q

@show a1' * reduce(*, [A[parse(Int, c) + 1] for c in reverse(string(7 * 99 + 7, base=2))]) * a∞
