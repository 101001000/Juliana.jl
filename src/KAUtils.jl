module KAUtils

    using KernelAbstractions
    using BenchmarkTools
    import GPUArrays.DataRef
    export ArrayConstructor, ktime

    macro comment(str)
        return :()
    end

    macro qtime(expr)

        backend_id = expr.args[1].args[2]

        return quote @btime begin esc($expr); KernelAbstractions.synchronize(backend_id) end evals=1 samples=1  end
    end

    macro ktime(kernel, backend)
        a = Expr(:call, Expr(:. , :KernelAbstractions, QuoteNode(:synchronize)), backend)
        ex = Expr(:block, kernel, a, Expr(:call, :sleep, 1))
        quote
            BenchmarkTools.@btime $ex evals=1 samples=1
        end
    end

    function ArrayConstructor(backend, arr)
        d_arr = KernelAbstractions.allocate(backend, eltype(arr), size(arr))
        KernelAbstractions.copyto!(backend, d_arr, arr)
        KernelAbstractions.synchronize(backend)
        return d_arr
    end

    function ArrayConstructor(backend, T::Type, ::UndefInitializer, dims...)
        d_arr = KernelAbstractions.allocate(backend, T, dims...)
        return d_arr
    end

    function array2tuple(a::Array)
        (a...,)
     end

    # Multiply two tuples (making scalars 1 dim tuples) elementwise, and if they have different size, return the rest of the elements of the biggest tuple unchanged.
    function tuple_mult(A, B)

        if isnothing(A)
            A = 1
        end

        if isnothing(B)
            B = 1
        end

        lA = length(A) == 1 ? [A[1]] : collect(A)
        lB = length(B) == 1 ? [B[1]] : collect(B)

        b = length(lA) >= length(lB) ? lA : lB
        s = length(lA) < length(lB) ? lA : lB

        if length(b) == 1
            return b[1] * s[1]
        end

        for i in eachindex(s)
            b[i] *= s[i]
        end

        return array2tuple(b)
    end



end

