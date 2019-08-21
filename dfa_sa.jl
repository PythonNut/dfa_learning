using Printf

include("util.jl")

mutable struct DFA
    n::Int
    s::Int
    q0::Int
    δ::Array{Int, 2}
    F::BitVector

    function DFA(q0::Vector{Int}, F::Vector{Int}, A::Vector{Array{Int, 2}})
        n = length(q0)
        s = length(A)
        q0i = findfirst(q0 .== 1)

        δ = Array{Int,2}(undef, s, n)
        for σ in 1:s
            for state in 1:n
                δ[σ, state] = findfirst(A[σ][state, :] .== 1)
            end
        end

        new(n, s, q0i, δ, F)
    end


    function DFA(n::Int, s::Int)
        q0i = 1

        δ = Array{Int,2}(undef, s, n)
        for σ in 1:s
            for state in 1:n
                δ[σ, state] = rand(1:n)
            end
        end
        F = rand(Bool, n)
        new(n, s, q0i, δ, F)
    end
end

function dfa_eval(dfa::DFA, seq)
    q = dfa.q0
    for σ in seq
        q = dfa.δ[σ+1, q]
    end
    return dfa.F[q]
end

function dfa_score(dfa::DFA, train::Vector{Any})
    tn, tp, fn, fp = 0, 0, 0, 0

    for (seq, accept) in train
        predict = dfa_eval(dfa, seq)
        if accept == 1
            if predict
                tp += 1
            else
                fn += 1
            end
        else
            if predict
                fp += 1
            else
                tn += 1
            end
        end
    end

    return tn, tp, fn, fp
end

function acc_score(tn, tp, fn, fp)
    t = tn + tp
    return t/(fn + fp + t)
end

function accept_prob(cur::Float64, new::Float64, temp::Float64)
    if new > cur
        return 1
    else
        return exp((new - cur)/temp)
    end
end


function anneal(n::Int, s::Int, train::Vector{Any})
    temp = 20/1050
    cooling_rate = 0.9999999
    cooling_rate = 0.99999999
    dfa = DFA(n, s)
    temp_min = 0.2/105
    fit = acc_score(dfa_score(dfa, train)...)
    best, best_fit = deepcopy(dfa), fit
    count = 0
    while temp > temp_min
        mutate_accept = false
        σ, i, save = 0, 0, 0
        if rand() > 1/n
            # change a random transition
            σ = rand(1:s)
            i = rand(1:n)
            j = rand(1:n)
            save = dfa.δ[σ, i]
            dfa.δ[σ, i] = j
        else
            mutate_accept = true
            # change a accept state
            i = rand(1:n)
            dfa.F[i] = !dfa.F[i]
        end

        new_fit = acc_score(dfa_score(dfa, train)...)
        ap = accept_prob(fit, new_fit, temp)
        if ap > rand()
            # keep it
            fit = new_fit
        else
            # undo the move
            if mutate_accept
                dfa.F[i] = !dfa.F[i]
            else
                dfa.δ[σ, i] = save
            end
        end

        if new_fit > best_fit
            best = deepcopy(dfa)
            best_fit = new_fit

            if best_fit == 1
                break
            end
        end

        if count % 500 == 0
            @printf("%.4f %.4f %f %f %s\n", new_fit, best_fit, temp, ap, dfa)
        end

        count += 1
        temp *= cooling_rate
    end

    return best
end
