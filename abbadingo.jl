function read_dct(fname)
    train = []
    open(fname) do file
        for (i, l) in enumerate(eachline(file))
            if i == 1
                continue
            end

            nums = parse.(Int, split(l, ' '))
            accept = nums[1]
            str = tuple(nums[3:end]...)
            push!(train, (str, accept))
        end
    end
    return train
end
