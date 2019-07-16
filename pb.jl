using Printf

function reversedims(ary::Array)
    permutedims(ary, ndims(ary):-1:1)
end

function parse_naps(fname, MOD, syms)
    assignments = cat([split(l, " ")[2:end] for l in readlines(fname) if l[1] == 'v']..., dims=1)
    results = Dict{Int, Bool}()
    max_key = 0
    for ass in assignments
        if ass[1] == '-'
            key = parse(Int, ass[3:end])
            results[key] = 0
        else
            key = parse(Int, ass[2:end])
            results[key] = 1
        end

        if key > max_key
            max_key = key
        end
    end

    flat = zeros(Int, max_key)
    for k in keys(results)
        flat[k] = results[k]
    end

    i = 1l
    arrays = []
    for s in syms
        l = length(MOD[s])
        push!(arrays, reversedims(reshape(flat[i:i+l-1], reverse(size(MOD[s])))))
        i += l
    end

    return arrays
end

function print_constraint(cref)
    s = string(cref)
    subs = [
        r"[[,]" => "_",
        "≥" => ">=",
        "≤" => "<=",
        "]" => "",
        r"([0-9]).0" => s"\1",
        r"-([a-z])" => s"-1 \1"
    ]
    for sub in subs
        s = replace(s, sub)
    end
    return s
end

function allocate_variables(shapes)
    i = 1
    results = []
    for shape in shapes
        size = reduce(*, shape)
        push!(results, reshape(i:i+size-1, shape))
        i += size
    end
    return results
end

function deallocate_variables(flat, shapes)
    i = 1
    arrays = []
    for shape in shapes
        size = reduce(*, shape)
        push!(arrays, reshape(flat[i:i+size-1], shape))
        i += size
    end
    return arrays
end

function pb_format_coeff(c)
    if c > 0
        return "+" * string(c)
    else
        return string(c)
    end
end


function pb_format_comb(cs, vs)
    @assert length(cs) == length(vs)
    join((pb_format_coeff(c) * " x" * string(v) for (c, v) in zip(cs, vs)), ' ')
end

function pb_format_sum(vs)
    pb_format_comb(repeat([1], length(vs)), vs)
end

function min_dfa_emit_pb(train, prefixes, G, Σ, h, io=Base.stdout)
    VV = vertices(G)
    EE = edges(G)
    n = nv(G)
    m = ne(G)
    s = length(Σ)

    x, w, y, z = allocate_variables([(n, h), (h,), (s, h, h), (h,)])

    # @objective(MOD, MOI.MIN_SENSE, sum(w))
    @printf(io, "min: %s ;\n", pb_format_sum(w))

    for v in VV
        # @constraint(MOD, sum(x[v,:]) == 1)
        @printf(io, "%s = 1 ;\n", pb_format_sum(x[v,:]))
    end

    for e in EE
        for i=1:h
            # @constraint(MOD, x[e.src,i] + x[e.dst,i] <= w[i])
            @printf(io, "%s <= 0 ;\n", pb_format_comb(
                [1, 1, -1],
                [x[e.src,i], x[e.dst,i], w[i]]
            ))
        end
    end

    for i in 1:h
        # @constraint(MOD, w[i] <= sum(x[:,i]))
        @printf(io, "%s <= 0 ;\n", pb_format_comb(
            [1, repeat([-1], n)...],
            [w[i], x[:,i]...]
        ))
    end

    for i in 2:h
        # @constraint(MOD, w[i] <= w[i-1])
        @printf(io, "%s <= 0 ;\n", pb_format_comb(
            [1, -1],
            [w[i], w[i-1]]
        ))
    end

    for (pi, p) in enumerate(prefixes)
        for l in 1:s
            ci = findfirst(isequal(tuple(p..., Σ[l])), prefixes)
            if !isnothing(ci)
                for j in 1:h
                    # @constraint(MOD, sum(y[l, j, :]) == w[j])
                    @printf(io, "%s = 0 ;\n", pb_format_comb(
                        [repeat([1], h)..., -1],
                        [y[l, j, :]..., w[j]]
                    ))

                    for k in 1:h
                        # @constraint(MOD, y[l, j, k] >= x[pi, j] + x[ci, k] - 1)
                        @printf(io, "%s <= 1 ;\n", pb_format_comb(
                            [1, 1, -1],
                            [x[pi, j], x[ci, k], y[l, j, k]]
                        ))

                        # @constraint(MOD, x[ci, k] >= x[pi, j] + y[l, j, k] - 1)
                        @printf(io, "%s <= 1 ;\n", pb_format_comb(
                            [1, 1, -1],
                            [x[pi, j], y[l, j, k], x[ci, k]]
                        ))
                    end
                end
            end
        end
    end

    for (seq, label) in train
        pi = findfirst(isequal(seq), prefixes)
        for i in 1:h
            if label == 1
                # @constraint(MOD, x[pi, i] <= z[i])
                @printf(io, "%s <= 0 ;\n", pb_format_comb(
                    [1, -1],
                    [x[pi, i], z[i]]
                ))
            else
                # @constraint(MOD, x[pi, i] <= 1-z[i])
                @printf(io, "%s <= 1 ;\n", pb_format_comb(
                    [1, 1],
                    [x[pi, i], z[i]]
                ))
            end
        end
    end

    flush(io)
end


function extract_dfa(x, w, y, z, Σ, prefixes)
    k = sum(w)

    A = [y[l, 1:k, 1:k] for l in 1:length(Σ)]
    q1 = x[findfirst(isequal(()), prefixes), 1:k]
    q∞ = z[1:k]

    return q1, q∞, A
end

function min_dfa_pb(train, prefixes, G, Σ, h)
    mktemp() do path, io
        min_dfa_emit_pb(train, prefixes, G, Σ, h, io)

        NAPS_PATH = `./naps/naps $(path)`

        inp = Pipe()
        out = Pipe()
        err = Pipe()
        process = run(pipeline(NAPS_PATH, stdin=inp, stdout=out, stderr=err), wait=false)

        close(out.in)
        close(err.in)
        close(inp)

        stdout = @async String(read(out))
        stderr = @async String(read(err))


        wait(process)

        assignments = cat([split(l, ' ')[2:end] for l in split(fetch(stdout), '\n') if length(l) > 1 && l[1] == 'v']..., dims=1)
        results = Dict{Int, Bool}()
        max_key = 0
        for ass in assignments
            if ass[1] == '-'
                key = parse(Int, ass[3:end])
                results[key] = 0
            else
                key = parse(Int, ass[2:end])
                results[key] = 1
            end

            if key > max_key
                max_key = key
            end
        end

        flat = zeros(Int, max_key)
        for k in keys(results)
            flat[k] = results[k]
        end

        s = length(Σ)
        n = nv(G)
        x, w, y, z = deallocate_variables(flat, [(n, h), (h,), (s, h, h), (h,)])
        return extract_dfa(x, w, y, z, Σ, prefixes)
    end
end
