

module KAUtils

    using KernelAbstractions
    import GPUArrays.DataRef
    export ArrayConstructor

    macro comment(str)
        return :()
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

    
end

