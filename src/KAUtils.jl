module KAUtils

    using KernelAbstractions
    using CUDA
    using AMDGPU
    using GPUArrays
    import GPUArrays.DataRef
    export ArrayConstructor, ktime

    macro comment(str)
        return :()
    end

    DeviceArray{T, A, N} = Union{CUDA.CuDeviceArray{T, A, N}, AMDGPU.Device.ROCDeviceArray{T, A, N}}

    struct Device
        ordinal::Integer
        function Device(ordinal::Integer)
            new(ordinal)
        end
    end

    function device()
        return Device(0)
    end

    function devices()
        return [device()]
    end

    function name(dev::Device)
        return "Un-implemented device name functionality"
    end

    function get_backend()

        backend_var = get(ENV, "KA_BACKEND", "CUDA")

        @info "Using backend " * backend_var

        if backend_var == "CUDA"
            return CUDABackend()
        elseif backend_var == "AMD"
            return ROCBackend()
        elseif backend_var == "oneAPIBackend"
            return oneAPIBackend()
        elseif backend_var == "MetalBackend"
            return MetalBackend()
        else   
            throw("backend " * backend_var * " not recognized")
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

    zeros(backend, T::Type, dims...) = KernelAbstractions.zeros(backend, T, dims...)
    zeros(backend, dims...) = KernelAbstractions.zeros(backend, Float32, dims...)
    ones(backend, T::Type, dims...) = KernelAbstractions.ones(backend, T, dims...)
    ones(backend, dims...) = KernelAbstractions.ones(backend, Float32, dims...)

    function fill(backend::Backend, v, dims...)
        data = KernelAbstractions.allocate(backend, typeof(v), dims...)
        fill!(data, v)
        return data
    end

    function array2tuple(a::Array)
        (a...,)
    end

    function default_rng(backend)
        if typeof(backend) == "CUDABackend"
            return GPUArrays.default_rng(CuArray)
        elseif typeof(backend) == "ROCBackend"
            return GPUArrays.default_rng(ROCArray)
        elseif typeof(backend) == "oneAPIBackend"
            return GPUArrays.default_rng(oneArray)
        elseif typeof(backend) == "MetalBackend"
            return GPUArrays.default_rng(MtlArray)
        else   
            throw("default_rng not implemented for this backend")
        end
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
            @error "free_memory not implemented for this backend"
            return default_free_memory
        end
    end


end

