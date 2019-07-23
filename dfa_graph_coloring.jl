using LightGraphs
using Printf
using DataStructures

include("util.jl")

mutable struct IndexDispenser
    offset::Int
    variables::Vector{Any}

    function IndexDispenser()
        new(0, [])
    end
end

function dispense(I::IndexDispenser, size)
    len = reduce(* , size)
    V = reshape((I.offset + 1):(I.offset + len), size)
    push!(I.variables, V)
    I.offset += len
    return V
end

function auto_add_edge!(G::SimpleGraph, src, dst)
    n = nv(G)
    if src > n || dst > n
        add_vertices!(G, max(src, dst) - n)
    end
    return add_edge!(G, src, dst)
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

function add_inequality_constraints!(G, v, train, prefixes, suffixes)
    dmap = DefaultDict{Int, DefaultDict{Bool, Set}}(()->DefaultDict{Bool, Set}(Set))

    for (seq, accept) in train
        for cut in 0:length(seq)
            prefix = seq[1:cut]
            suffix = seq[cut+1:end]
            pi = findfirst!(isequal(prefix), prefixes)
            si = findfirst!(isequal(suffix), suffixes)
            push!(dmap[si][accept], pi)
        end
    end

    for suffix in values(dmap)
        if true in keys(suffix) && false in keys(suffix)
            for (p0, p1) in IterTools.product(suffix[true], suffix[false])
                auto_add_edge!(G, v[p0], v[p1])
            end
        end
    end
    return dmap
end

function add_clique!(G, clique)
    for (i, j) in subsets(clique,2)
        auto_add_edge!(G, i, j)
    end
end

function add_multi_edge!(G, us, vs)
    for (u, v) in Iterators.product(us, vs)
        auto_add_edge!(G, u, v)
    end
end


function main()
    s = 2
    Σ = collect(0:s-1)
    h = 12

    train = read_dct("dcts/dfa_8_try_1.dct")
    @printf("Given %d training examples\n", length(train))

    prefixes, suffixes = enumerate_fixes(train)
    @printf("Extracted %d prefixes\n", length(prefixes))

    G = SimpleGraph()

    I = IndexDispenser()
    v = dispense(I, size(prefixes))

    q = dispense(I, (h,))
    add_clique!(G, q)

    t = dispense(I, (h, s))

    for (ui, prefix) in enumerate(prefixes)
        for l in 1:s
            xi = findfirst(isequal(tuple(prefix..., Σ[l])), prefixes)
            if !isnothing(xi)
                for (q, t) in zip(q, t[:,l])
                    c1 = dispense(I, (h-2,))
                    c2 = dispense(I, (h-2,))
                    c3 = dispense(I, (h-2,))
                    aux1, aux2, xpp, xp = dispense(I, (4,))
                    x = v[xi]
                    u = v[ui]

                    add_clique!(G, c1)
                    add_clique!(G, c2)
                    add_clique!(G, c3)

                    # Add top edges
                    auto_add_edge!(G, u, xpp)
                    auto_add_edge!(G, xpp, xp)
                    auto_add_edge!(G, xp, x)

                    # Add bottom edges
                    auto_add_edge!(G, q, aux1)
                    auto_add_edge!(G, aux1, aux2)
                    auto_add_edge!(G, aux2, t)

                    # Add clique connections
                    add_multi_edge!(G, [xpp, q, aux1], c1)
                    add_multi_edge!(G, [xp, aux2, aux1], c2)
                    add_multi_edge!(G, [x, t, aux2], c3)
                end
            end
        end
    end

    return G
end

function write_dimacs(G::SimpleGraph, fname)
    open(fname, "w") do io
        println(io, "p edge $(nv(G)) $(ne(G))")
        for e in edges(G)
            println(io, "e $(e.src) $(e.dst)")
        end
    end

end
