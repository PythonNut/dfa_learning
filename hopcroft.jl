function hopcropft_equivalence_classes(Q, Σ, δ, q∞)
    F = Set(findall(q∞ .> 0))
    Q = Set(collect(Q))

    P = Set([F, setdiff(Q, F)])
    W = Set([F])

    while !isempty(W)
        A = pop!(W)
        for c in Σ
            X = Set([
                q for q in Q
                if findfirst(δ[c][q,:] .== 1) ∈ A
            ])
            for Y in P
                I = intersect(X, Y)
                D = setdiff(Y, X)
                if isempty(I) || isempty(D)
                    continue
                end
                pop!(P, Y)
                push!(P, I)
                push!(P, D)
                if Y ∈ W
                    pop!(W, Y)
                    push!(W, I)
                    push!(W, D)
                else
                    if length(I) <= length(D)
                        push!(W, I)
                    else
                        push!(W, D)
                    end
                end
            end
        end
    end

    return P
end
