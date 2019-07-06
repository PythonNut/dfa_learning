using DataStructures
using UniqueVectors
using IterTools
# using SimpleGraphs
using LightGraphs
using Gurobi
using Cbc
using JuMP
using Printf

# Section 1: Define the DFA

const s = 2
const d = 7

function array_mod(a, base, mod)
    result = 0
    for x in a
        result *= base
        result += x
        result %= mod
    end
    return result
end


function dfa(x)
    return Int(array_mod(x, 2, d) == 0) + Int(array_mod(x, 2, d) == 4)
end

# Section 2: Build the training set

train = []

for l in 1:10
    for seq in Iterators.product(repeat([collect(0:s-1)], l)...)
        if randn() > 1.0
            push!(train, (seq, dfa(seq)))
        end
    end
end


# Section 3: Build the prefix map

prefix_map = DefaultDict{Int, Dict{Int, Int}}(Dict)
suffix_map = DefaultDict{Int, Dict{Int, Int}}(Dict)

prefixes = UniqueVector([])
suffixes = UniqueVector([])

for (seq, result) in train
    for cut in 0:length(seq)
        prefix = seq[1:cut]
        suffix = seq[cut+1:end]
        pi = findfirst!(isequal(prefix), prefixes)
        si = findfirst!(isequal(suffix), suffixes)
        @assert length(prefix) + length(suffix) <= 10
        prefix_map[pi][si] = result
        suffix_map[si][pi] = result
        @assert length(suffixes[si]) + length(prefixes[pi]) <= 10
    end
end

num_p, num_s = length(prefixes), length(suffixes)

mask = falses(num_p, num_s)
dmap = BitArray(undef, num_p, num_s)

for (seq, result) in train
    for cut in 0:length(seq)
        prefix = seq[1:cut]
        suffix = seq[cut+1:end]
        pi = findfirst(isequal(prefix), prefixes)
        si = findfirst(isequal(suffix), suffixes)
        dmap[pi, si] = result
        mask[pi, si] = true
    end
end

# Section 4: Build the distinguishability graph

# G = IntGraph(num_p)
G = SimpleGraph(num_p)

for (i, j) in subsets(1:num_p, 2)
    if any((mask[i, :] .& mask[j, :]).*(dmap[i, :] .!= dmap[j, :]))
        add_edge!(G, i, j)
    end
end

# Section 5: Color the graph

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

# Try generating suffix constraints
suffix_ϵ = findfirst(isequal(()), suffixes)
suffix_sets = Dict(k=>v for (k, v) in suffix_map if length(v) > 1 && k != suffix_ϵ)
suffix_keys = UniqueVector(keys(suffix_sets))

# clique = max_clique(G)
cliques = maximal_cliques(G)
clique = cliques[findmax(length.(cliques))[2]]

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

# MOD = Model(with_optimizer(Gurobi.Optimizer))
MOD = Model(with_optimizer(Gurobi.Optimizer, Method=2, InfUnbdInfo=1, NumericFocus=2))
# MOD = Model(with_optimizer(Cbc.Optimizer; logLevel=3, threads=Sys.CPU_THREADS))
# MOD = Model(with_optimizer(Cbc.Optimizer; logLevel=3))

@variable(MOD, x[VV,1:h], Bin)
@variable(MOD, w[1:h], Bin)
@variable(MOD, y[1:length(suffix_keys), 1:h, 1:h], Bin)
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

for (i, si) in enumerate(suffix_keys)
    for pi in keys(suffix_sets[si])
        for j in 1:h
            @constraint(MOD, sum(y[i, j, :]) == 1)
            for k in 1:h
                fi = findfirst(isequal(tuple(prefixes[pi]..., suffixes[si]...)), prefixes)
                @constraint(MOD, y[i, j, k] >= x[pi, j] + x[fi, k] - 1)
                @constraint(MOD, x[fi, k] >= x[pi, j] + y[i, j, k] - 1)
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


println("Starting optimizer")

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

k = round(Int, objective_value(MOD))

A0 = round.(Int, value.(y[findfirst(isequal(findfirst(isequal((0,)), suffixes)), suffix_keys),1:k,1:k]))
A1 = round.(Int, value.(y[findfirst(isequal(findfirst(isequal((1,)), suffixes)), suffix_keys),1:k,1:k]))
q1 = round.(Int, value.(x[findfirst(isequal(()), prefixes),1:k]))
q∞ = round.(Int, value.(z[1:k]))

function int2bitarray(i)
    return [parse(Int, c) for c in string(i, base=2)]
end

function wfa_eval(a1, a∞, A, s)
    return reduce(*, [a1', (A[c + 1] for c in s)..., a∞])
end

@show [wfa_eval(q1, q∞, [A0, A1], int2bitarray(d*99^6 + k)) for k in 0:d-1]
