using JuMP
using ProxSDP
using LowRankApprox
using LinearAlgebra
using Clustering
using Statistics

include("util.jl")

function learn_dfa(fname, h=50)
    s = 2
    Σ = collect(0:s-1)

    train = read_dct(fname)
    @printf("Given %d training examples\n", length(train))

    prefixes, suffixes = enumerate_fixes(train)
    @printf("Extracted %d prefixes\n", length(prefixes))

    hankel, mask = partial_hankel(train, prefixes, suffixes)

    while true
        model = Model(with_optimizer(ProxSDP.Optimizer, log_verbose=true))
        n, m = size(hankel)

        @variable(model, W1[1:n, 1:n])
        @variable(model, W2[1:m, 1:m])
        @variable(model, X[1:n, 1:m])

        @constraint(model, [W1 X; X' W2] in PSDCone())
        @constraint(model, mask .* (X - hankel) .== 0)
        @c(model, X .>= 0)

        @objective(model, MOI.MIN_SENSE, tr(W1) + tr(W2))

        optimize!(model)

        prev_mask_nnz = sum(mask)

        for (i, x) in enumerate(value.(X))
            if abs(x) < 1e-5
                hankel[i] = false
                mask[i] = true
            end
        end

        mask_nnz = sum(mask)

        if prev_mask_nnz == mask_nnz
            break
        end
    end
end
