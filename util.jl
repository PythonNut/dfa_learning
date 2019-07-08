using LightGraphs
using LightGraphs.Parallel
using DataStructures
using UniqueVectors
using IterTools
using Printf

function array_mod(a, base, mod)
    result = 0
    for x in a
        result *= base
        result += x
        result %= mod
    end
    return result
end

function int2bitarray(i)
    return [parse(Int, c) for c in string(i, base=2)]
end

function wfa_eval(a1, a∞, A, s)
    return reduce(*, [a1', (A[c + 1] for c in s)..., a∞])
end

function min_color(G::SimpleGraph, h=LightGraphs.random_greedy_color(G, 100).num_colors)
    VV = vertices(G)
    EE = edges(G)
    n = nv(G)
    m = ne(G)

    err_msg = "This graph is not $(h) colorable"

    # MOD = Model(with_optimizer(Cbc.Optimizer; logLevel=1, threads=Sys.CPU_THREADS))
    MOD = Model(with_optimizer(Gurobi.Optimizer))
    # MOD = Model(with_optimizer(GLPK.Optimizer, msg_lev=GLPK.MSG_ON))

    @variable(MOD, x[VV,1:h], Bin)
    @variable(MOD, w[1:h], Bin)

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

    @objective(MOD, MOI.MIN_SENSE, sum(w))

    optimize!(MOD)
    status = Int(termination_status(MOD))

    if status != 1
        error(err_msg)
    end

    X = value.(x)

    result = Dict{Int,Int}()

    for v in VV
        for c = 1:h
            if X[v,c] > 0
                result[v] = c
            end
        end
    end

    return result
end

function max_indep_set(G::SimpleGraph)
    VV = vertices(G)
    EE = edges(G)
    n = nv(G)
    m = ne(G)

    MOD = Model(with_optimizer(Gurobi.Optimizer))
    # MOD = Model(with_optimizer(Cbc.Optimizer; logLevel=3, threads=Sys.CPU_THREADS))
    @variable(MOD, x[VV],Bin)
    for e in EE
        u,v = e.src, e.dst
        @constraint(MOD,x[u]+x[v]<=1)
    end
    @objective(MOD,Max,sum(x[v] for v in VV))
    optimize!(MOD)

    X = value.(x)
    A = Set([v for v in VV if X[v]>0.1])

    return A
end

function max_clique(G::SimpleGraph)
    return max_indep_set(LightGraphs.complement(G))
end


function max_clique2(G)
    cliques = maximal_cliques(G)
    clique = cliques[findmax(length.(cliques))[2]]
    return clique
end

function min_color3(G::SimpleGraph)
    VV = vertices(G)
    EE = edges(G)
    n = nv(G)
    m = ne(G)

    gc = LightGraphs.random_greedy_color(G, 100)
    h = gc.num_colors

    dgc = LightGraphs.degree_greedy_color(G)
    if dgc.num_colors <= h
        h = dgc.num_colors
    end

    err_msg = "This graph is not $(h) colorable"

    MOD = Model(with_optimizer(Gurobi.Optimizer))

    @variable(MOD, x[VV,1:h], Bin)
    @variable(MOD, w[1:h], Bin)

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

    @objective(MOD, MOI.MIN_SENSE, sum(w))

    # clique = max_clique(G)
    cliques = maximal_cliques(G)
    clique = cliques[findmax(length.(cliques))[2]]
    for (c, v) in enumerate(clique)
        @constraint(MOD, x[v, c] == 1)
    end


    optimize!(MOD)
    status = Int(termination_status(MOD))

    if status != 1
        error(err_msg)
    end

    X = value.(x)

    result = Dict{Int,Int}()

    for v in VV
        for c = 1:h
            if X[v,c] > 0
                result[v] = c
            end
        end
    end

    return result
end

function smart_greedy_color(G, n=1000)
    gc = Parallel.random_greedy_color(G, n)

    dgc = LightGraphs.degree_greedy_color(G)
    if dgc.num_colors <= gc.num_colors
        gc = dgc
    end
    return gc
end
