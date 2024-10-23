module KAUtils

    using KernelAbstractions
    using CUDA
    import GPUArrays.DataRef
    export ArrayConstructor, ktime

    macro comment(str)
        return :()
    end

    #macro qtime(expr)
#
    #    backend_id = expr.args[1].args[2]
#
    #    return quote @btime begin esc($expr); KernelAbstractions.synchronize(backend_id) end evals=1 samples=1  end
    #end
#
    #macro ktime(kernel, backend)
    #    a = Expr(:call, Expr(:. , :KernelAbstractions, QuoteNode(:synchronize)), backend)
    #    ex = Expr(:block, kernel, a, Expr(:call, :sleep, 1))
    #    quote
    #        BenchmarkTools.@btime $ex evals=1 samples=1
    #    end
    #end

    function get_backend()
        return CUDABackend()
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

    zeros(backend, T::Type, dims...) = KernelAbstractions.zeros(backend, T, dims...)
    zeros(backend, dims...) = KernelAbstractions.zeros(backend, Float32, dims...)
    ones(backend, T::Type, dims...) = KernelAbstractions.ones(backend, T, dims...)
    ones(backend, dims...) = KernelAbstractions.ones(backend, Float32, dims...)

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

    function free_memory(backend)
        #TODO: fill this.
        default_free_memory = 4096 * 2^20

        if typeof(backend) == "CUDABackend"
            return CUDA.free_memory()
        elseif typeof(backend) == "ROCBackend"
            #AMD.free()
        elseif typeof(backend) == "oneAPIBackend"

        elseif typeof(backend) == "MetalBackend"

        elseif typeof(backend) == "CPU"
            return Sys.free_memory()
        else
            return default_free_memory
        end
    end


end

