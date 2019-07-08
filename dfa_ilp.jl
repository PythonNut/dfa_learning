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

function min_dfa_setup_model(MOD, train, prefixes, G, Σ, h, clique)
    VV = vertices(G)
    EE = edges(G)
    n = nv(G)
    m = ne(G)


    @variable(MOD, x[VV,1:h], Bin)
    @variable(MOD, w[1:h], Bin)
    @variable(MOD, y[Σ, 1:h, 1:h], Bin)
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
        for σ in Σ
            ci = findfirst(isequal(tuple(p..., σ)), prefixes)
            if !isnothing(ci)
                for j in 1:h
                    @constraint(MOD, sum(y[σ, j, :]) == 1)
                    for k in 1:h
                        @constraint(MOD, y[σ, j, k] >= x[pi, j] + x[ci, k] - 1)
                        @constraint(MOD, x[ci, k] >= x[pi, j] + y[σ, j, k] - 1)
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


    @objective(MOD, MOI.MIN_SENSE, sum(w))

    for (c, v) in enumerate(clique)
        @constraint(MOD, x[v, c] == 1)
    end
end

function solve_min_dfa_model(MOD, Σ, prefixes)
    optimize!(MOD)
    status = Int(termination_status(MOD))

    if status != 1
        # TODO: Make this error message better
        error("ERROR")
    end

    k = round(Int, objective_value(MOD))

    A = [round.(Int, value.(collect(MOD[:y][σ, :, :])[1:k, 1:k])) for σ in Σ]
    q1 = round.(Int, value.(MOD[:x][findfirst(isequal(()), prefixes), 1:k]))
    q∞ = round.(Int, value.(MOD[:z][1:k]))

    return q1, q∞, A
end

function main()
    s = 2
    d = 17
    Σ = collect(0:s-1)

    function dfa(x)
        return Int(array_mod(x, 2, d) == 0) + Int(array_mod(x, 2, d) == 4)
    end

    train = random_iid_training_set(dfa, Σ, 10, 0.4)
    @printf("Given %d training examples\n", length(train))

    prefixes, suffixes = enumerate_fixes(train)
    @printf("Extracted %d prefixes\n", length(prefixes))

    hankel, mask = partial_hankel(train, prefixes, suffixes)
    G = build_distinguishability_graph(hankel, mask)

    @printf("Found %d inequality constraints\n", ne(G))

    clique = max_clique(G)
    @printf("Found inequality clique of size %d\n", length(clique))


    h = smart_greedy_color(G).num_colors
    @printf("Found greedy coloring with %d colors\n", h)

    MOD = Model(with_optimizer(Gurobi.Optimizer))
    # MOD = Model(with_optimizer(CPLEX.Optimizer))
    # MOD = Model(with_optimizer(Mosek.Optimizer))
    # MOD = Model(with_optimizer(Gurobi.Optimizer, Method=2, InfUnbdInfo=1, NumericFocus=2, BarConvTol=1e-12, CrossoverBasis=1))
    # MOD = Model(with_optimizer(Cbc.Optimizer; logLevel=3, threads=Sys.CPU_THREADS))
    # MOD = Model(with_optimizer(Cbc.Optimizer; logLevel=3))

    min_dfa_setup_model(MOD, train, prefixes, G, Σ, h, clique)

    q1, q∞, A = solve_min_dfa_model(MOD, Σ, prefixes)

    @show [wfa_eval(q1, q∞, A, int2bitarray(d*99^6 + k)) for k in 0:d-1]
end
