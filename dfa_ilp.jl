# using SimpleGraphs
using Gurobi
using CPLEX
using Cbc
using Mosek
using MosekTools
using JuMP

include("util.jl")

# Section 1: Define the DFA

function random_iid_training_set(dfa, Σ, max_len, p)
    train = []

    for l in 0:max_len
        for seq in Iterators.product(repeat([Σ], l)...)
            if rand() < p
                push!(train, (seq, dfa(seq)))
            end
        end
    end

    return train
end

function enumerate_fixes(train)
    prefixes = UniqueVector([])
    suffixes = UniqueVector([])

    for (seq, result) in train
        for cut in 0:length(seq)
            prefix = seq[1:cut]
            suffix = seq[cut+1:end]
            pi = findfirst!(isequal(prefix), prefixes)
            si = findfirst!(isequal(suffix), suffixes)
        end
    end
    return prefixes, suffixes
end

function partial_hankel(train, prefixes, suffixes)
    num_p, num_s = length(prefixes), length(suffixes)

    mask = falses(num_s, num_p)
    dmap = BitArray(undef, num_s, num_p)

    for (seq, result) in train
        for cut in 0:length(seq)
            prefix = seq[1:cut]
            suffix = seq[cut+1:end]
            pi = findfirst(isequal(prefix), prefixes)
            si = findfirst(isequal(suffix), suffixes)
            dmap[si, pi] = result
            mask[si, pi] = true
        end
    end

    return dmap, mask
end

function build_distinguishability_graph(dmap, mask)
    num_p = size(dmap)[2]
    G = SimpleGraph(num_p)

    for (i, j) in subsets(1:num_p, 2)
        if any((mask[:, i] .& mask[:, j]).*(dmap[:, i] .!= dmap[:, j]))
            add_edge!(G, i, j)
        end
    end
    return G
end

function min_dfa_setup_model!(MOD, train, prefixes, G, Σ, h)
    # standard model setup
    VV = vertices(G)
    EE = edges(G)
    n = nv(G)
    m = ne(G)
    s = length(Σ)

    @variable(MOD, x[VV,1:h], Bin)
    @variable(MOD, w[1:h], Bin)
    @variable(MOD, y[1:s, 1:h, 1:h], Bin)
    @variable(MOD, z[1:h], Bin)

    # for i in 1:h
    #     set_start_value(w[i], true)
    # end

    # for (v, c) in zip(VV, gc.colors)
    #     for i in 1:h
    #         set_start_value(x[v, i], i==c)
    #     end
    # end

    for v in VV
        @constraint(MOD, sum(x[v,:]) == 1)
    end

    for e in EE
        for i=1:h
            @constraint(MOD, x[e.src,i] + x[e.dst,i] <= w[i])
        end
    end

    for i in 1:h
        @constraint(MOD, w[i] <= sum(x[:,i]))
    end

    for i in 2:h
        @constraint(MOD, w[i] <= w[i-1])
    end

    for (pi, p) in enumerate(prefixes)
        for l in 1:s
            ci = findfirst(isequal(tuple(p..., Σ[l])), prefixes)
            if !isnothing(ci)
                for j in 1:h
                    for k in 1:h
                        @constraint(MOD, y[l, j, k] - x[ci, k] >= x[pi, j] - 1)
                        @constraint(MOD, x[ci, k] - y[l, j, k] >= x[pi, j] - 1)
                    end
                end
            end
        end
    end

    for l in 1:s
        for j in 1:h
            @constraint(MOD, sum(y[l, j, :]) == w[j])
        end
    end

    for (seq, label) in train
        pi = findfirst(isequal(seq), prefixes)
        for i in 1:h
            if label == 1
                @constraint(MOD, x[pi, i] <= z[i])
            else
                @constraint(MOD, x[pi, i] <= 1-z[i])
            end
        end
    end

    @objective(MOD, MOI.MIN_SENSE, sum(w))
end

function min_dfa_setup_model2!(MOD, train, prefixes, G, Σ, h)
    # model setup with fixed number of states, does not generate w
    VV = vertices(G)
    EE = edges(G)
    n = nv(G)
    m = ne(G)
    s = length(Σ)

    @variable(MOD, x[VV,1:h], Bin)
    @variable(MOD, y[1:s, 1:h, 1:h], Bin)
    @variable(MOD, z[1:h], Bin)

    for v in VV
        @constraint(MOD, sum(x[v,:]) == 1)
    end

    for e in EE
        for i=1:h
            @constraint(MOD, x[e.src,i] + x[e.dst,i] <= 1)
        end
    end

    for (pi, p) in enumerate(prefixes)
        for l in s:1
            ci = findfirst(isequal(tuple(p..., Σ[l])), prefixes)
            if !isnothing(ci)
                for j in 1:h
                    @constraint(MOD, sum(y[l, j, :]) == 1)
                    for k in 1:h
                        @constraint(MOD, y[l, j, k] - x[ci, k] >= x[pi, j] - 1)
                        @constraint(MOD, x[ci, k] - y[l, j, k] >= x[pi, j] - 1)
                    end
                end
            end
        end
    end

    for (seq, label) in train
        pi = findfirst(isequal(seq), prefixes)
        for i in 1:h
            if label == 1
                @constraint(MOD, x[pi, i] <= z[i])
            else
                @constraint(MOD, x[pi, i] <= 1-z[i])
            end
        end
    end
end

function break_dfa_symmetry_max_clique!(MOD, clique)
    for (c, v) in enumerate(clique)
        @constraint(MOD, MOD[:x][v, c] == 1)
    end
end

function break_dfa_symmetry_bfs!(MOD, Σ, prefixes, h)
    y, w = MOD[:y], MOD[:w]
    s = length(Σ)

    @variable(MOD, p[1:h, 1:h], Bin)
    @variable(MOD, t[1:h, 1:h], Bin)

    ϵi = findfirst(isequal(()), prefixes)
    @constraint(MOD, MOD[:x][ϵi, 1] == 1)

    for j in 2:h
        # @constraint(MOD, 1 <= sum(p[j, 1:j-1]))
        @constraint(MOD, w[j] <= sum(p[j, 1:j-1]))
    end

    for (i, j) in subsets(1:h, 2)
        @constraint(MOD, 0 <= s*t[i, j] - sum(y[1:s, i, j]) <= s-1)
        @constraint(MOD, 1-i <= t[i, j] - sum(t[1:i-1, j]) - i*p[j,i] <= 0)
    end

    for (k, i, j) in subsets(1:h-1, 3)
        @constraint(MOD, p[j, i] <= 1-p[j+1, k])
    end

    if length(Σ) == 2
        for (i, j) in subsets(1:h-1, 2)
            @constraint(MOD, p[j, i] + p[j+1, i] - 1 <= y[1, i, j])
        end
    else
        @variable(MOD, m[1:s, 1:h, 1:h], Bin)

        for (i, j) in subsets(1:h, 2)
            for l in 1:s
                @constraint(MOD, 1-l <= y[l, i, j] - sum(y[1:l-1, i, j]) - l*m[l, i, j] <= 0)
            end

            for (l1, l2) in subsets(1:s, 2)
                @constraint(MOD, m[l2, i, j] <= 1-y[l1, i, j])
            end
        end

        for (i, j) in subsets(1:h-1, 2)
            for (l1, l2) in subsets(1:s, 2)
                @constraint(MOD, p[j, i] + p[j+1, i] + m[l2, i, j] - 2 <= 1 - m[l1, i, j+1])
            end
        end
    end
end

function break_dfa_symmetry_bfs2!(MOD, Σ, prefixes, h)
    y, w = MOD[:y], MOD[:w]
    s = length(Σ)

    @variable(MOD, p[1:h, 1:h], Bin)
    @variable(MOD, t[1:h, 1:h], Bin)

    ϵi = findfirst(isequal(()), prefixes)
    @constraint(MOD, MOD[:x][ϵi, 1] == 1)

    for j in 2:h
        # @constraint(MOD, 1 <= sum(p[j, 1:j-1]))
        @constraint(MOD, w[j] <= sum(p[j, 1:j-1]))
    end

    for (i, j) in subsets(1:h, 2)
        # @constraint(MOD, 0 <= s*t[i, j] - sum(y[1:s, i, j]) <= s-1)
        @constraint(MOD, t[i, j] <= sum(y[1:s, i, j]))
        @constraint(MOD, y[1:s, i, j] .<= t[i, j])

        # @constraint(MOD, 1-i <= t[i, j] - sum(t[1:i-1, j]) - i*p[j,i] <= 0)
        @constraint(MOD, p[j, i] <= t[i, j])
        @constraint(MOD, t[i, j] - sum(t[1:i-1, j]) <= p[j, i])
    end

    for (k, i, j) in subsets(1:h-1, 3)
        @constraint(MOD, p[j, i] <= 1-p[j+1, k])
    end

    for (k, i, j) in subsets(1:h, 3)
        @constraint(MOD, p[j, i] <= 1-t[k, j])
    end

    if length(Σ) == 2
        for (i, j) in subsets(1:h-1, 2)
            @constraint(MOD, p[j, i] + p[j+1, i] - 1 <= y[1, i, j])
        end
    else
        @variable(MOD, m[1:s, 1:h, 1:h], Bin)

        for (i, j) in subsets(1:h, 2)
            for l in 1:s
                @constraint(MOD, 1-l <= y[l, i, j] - sum(y[1:l-1, i, j]) - l*m[l, i, j] <= 0)
            end

            for (l1, l2) in subsets(1:s, 2)
                @constraint(MOD, m[l2, i, j] <= 1-y[l1, i, j])
            end
        end

        for (i, j) in subsets(1:h-1, 2)
            for (l1, l2) in subsets(1:s, 2)
                @constraint(MOD, p[j, i] + p[j+1, i] + m[l2, i, j] - 2 <= 1 - m[l1, i, j+1])
            end
        end
    end
end

function solve_min_dfa_model(MOD, Σ, prefixes, k=round(Int, objective_value(MOD)))
    status = Int(termination_status(MOD))

    if status != 1
        # TODO: Make this error message better
        error("ERROR")
    end

    A = [round.(Int, value.(MOD[:y][l, 1:k, 1:k])) for l in 1:length(Σ)]
    q1 = round.(Int, value.(MOD[:x][findfirst(isequal(()), prefixes), 1:k]))
    q∞ = round.(Int, value.(MOD[:z][1:k]))

    return q1, q∞, A
end

function main()
    s = 2
    d = 17
    Σ = collect(0:s-1)

    # function dfa(x)
    #     return Int(array_mod(x, 2, d) == 0) + Int(array_mod(x, 2, d) == 4)
    # end

    function dfa(x)
       if length(x) < 4
           return 0
       end

       return Int(x[end-3] == 1)
    end

    train = random_iid_training_set(dfa, Σ, 9, 0.6)
    @printf("Given %d training examples\n", length(train))

    prefixes, suffixes = enumerate_fixes(train)
    @printf("Extracted %d prefixes\n", length(prefixes))

    hankel, mask = partial_hankel(train, prefixes, suffixes)
    G = build_distinguishability_graph(hankel, mask)

    @printf("Found %d inequality constraints\n", ne(G))

    # clique = max_clique2(G)
    # @printf("Found inequality clique of size %d\n", length(clique))


    h = smart_greedy_color(G).num_colors
    @printf("Found greedy coloring with %d colors\n", h)

    # MOD = Model(with_optimizer(Gurobi.Optimizer, MIPFocus=1))
    MOD = Model(with_optimizer(NaPS.Optimizer))
    # MOD = direct_model(Gurobi.Optimizer(Method=2, InfUnbdInfo=1))
    # MOD = direct_model(Gurobi.Optimizer(Method=2, MIPFocus=1, InfUnbdInfo=1, DegenMoves=0))
    # MOD = Model(with_optimizer(CPLEX.Optimizer))
    # MOD = Model(with_optimizer(Mosek.Optimizer))
    # MOD = Model(with_optimizer(Gurobi.Optimizer, Method=2, InfUnbdInfo=1, NumericFocus=2, BarConvTol=1e-12, CrossoverBasis=1))
    # MOD = Model(with_optimizer(Cbc.Optimizer; logLevel=3, threads=Sys.CPU_THREADS))
    # MOD = Model(with_optimizer(Cbc.Optimizer; logLevel=3))

    min_dfa_setup_model!(MOD, train, prefixes, G, Σ, 50)
    break_dfa_symmetry_bfs!(MOD, Σ, prefixes, 50)

    # break_dfa_symmetry_max_clique!(MOD, clique)

    optimize!(MOD)

    q1, q∞, A = solve_min_dfa_model(MOD, Σ, prefixes)

    @show [wfa_eval(q1, q∞, A, int2bitarray(d*99^6 + k)) for k in 0:d-1]
    return q1, q∞, A
end

function learn_dfa(fname, h=50, solver=:naps, opbname=nothing)
    s = 2
    Σ = collect(0:s-1)

    train = read_dct(fname)
    @printf("Given %d training examples\n", length(train))

    prefixes, suffixes = enumerate_fixes(train)
    @printf("Extracted %d prefixes\n", length(prefixes))

    hankel, mask = partial_hankel(train, prefixes, suffixes)
    G = build_distinguishability_graph(hankel, mask)

    @printf("Found %d inequality constraints\n", ne(G))

    MOD = Model(with_optimizer(NaPS.Optimizer, solver, opbname))

    min_dfa_setup_model!(MOD, train, prefixes, G, Σ, h)
    break_dfa_symmetry_bfs!(MOD, Σ, prefixes, h)

    @show MOD

    optimize!(MOD)

    return solve_min_dfa_model(MOD, Σ, prefixes)
end
