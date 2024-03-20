import MacroTools
import CUDA

include("exprutils.jl")
include("warnings.jl")

function_call_id = Symbol("") # A terrible global variable which keeps track of the function id.

function extract_and_empty_kernel_ass!(expr)
    if expr isa Expr
        if expr.head == Symbol("=")
            if expr.args[1] == Symbol("kernel_call")
                new_expr = copy(expr)
                expr.head = :block  
                expr.args = []
                return new_expr
            end
        end
        for i in eachindex(expr.args)
            ass = extract_and_empty_kernel_ass!(expr.args[i])
            if !isnothing(ass)
                return ass
            end
        end
    end
    return nothing
end

function extract_kernel_name_from_call(expr)
    for arg in expr.args
        if arg isa Expr
            if arg.head == :call
                if arg.args[1] isa Symbol
                    return arg.args[1]
                else # in case is an interpolated kernel call
                    return arg.args[1].args[1]
                end
            end
        end
    end
    display("ERROR EXTRACTING THE NAME FROM THE KERNEL")
end


function extract_kernel_names!(expr, ids)
    if typeof(expr) != Expr
        return
    end
    if expr_identify_1(expr, """CUDA.var\"@cuda\"""")
        push!(ids, extract_kernel_name_from_call(expr))
        return
    end
    for i in eachindex(expr.args)
        extract_kernel_names!(expr.args[i], ids)   
    end
end





function warning_generator(expr)
    if typeof(expr) == Expr
        for arg in expr.args
            if arg == Symbol("CUDA") || arg == Symbol("NVTX") 
                emit_warning(UntranslatedWarning(expr_to_string(expr)))
            end
            warning_generator(arg)
        end
    end
end

function expression_replacer!(expr)

    try
        if expr.head == :call
            function_call_id = expr.args[1]
        end
    catch
    end

    for i in eachindex(expr.args)

        arg = expr.args[i]

        if expr.head == Symbol("import")
            continue
        end

        new_expr = expr_replacer(arg)

        if !isnothing(new_expr)
            expr.args[i] = new_expr
        end

        if typeof(expr.args[i]) == Expr
            expression_replacer!(expr.args[i])
        end 

    end
end


function symbol_replace!(expr, old_symbol, new_symbol)
    if typeof(expr) == Expr
        for i in eachindex(expr.args)
            if typeof(expr.args[i]) == QuoteNode
                if expr.args[i].value == Symbol(old_symbol)
                    expr.args[i] = QuoteNode(Symbol(new_symbol))
                end
            end
            if expr.args[i] == Symbol(old_symbol)
                expr.args[i] = Symbol(new_symbol)
            end
            symbol_replace!(expr.args[i], old_symbol, new_symbol)
        end
    end
end


function expr_replace!(expr, old_expr, new_expr)
    if typeof(expr) == Expr
        for i in eachindex(expr.args)
            if expr_to_string(expr.args[i]) == expr_to_string(old_expr)
                expr.args[i] = new_expr
                continue
            end
            expr_replace!(expr.args[i], old_expr, new_expr)
        end
    end
end

function generate_kernel_call(expr)

    @assert expr.head == :macrocall

    fun_call = nothing
    threads = nothing
    blocks = nothing

    for arg in expr.args
        if typeof(arg) != Expr
            continue
        end
        if arg.head == Symbol("call")
            fun_call = arg
        end
        if arg.head == Symbol("=")
            for i in eachindex(arg.args)
                subarg = arg.args[i]
                if subarg == Symbol("threads") && isnothing(threads) 
                    threads = arg.args[i+1]
                    try #TODO: Magic number here, maybe parametrize the max threadsize warning?
                        eval(remove_interpolation(Expr(Symbol("="), :threads_value, arg.args[i+1])))
                        if prod(threads_value) > 256
                            emit_warning(ThreadSizeTooLarge())
                        end
                    catch
                        emit_warning(ThreadSizeNotChecked())
                    end
                end
                if subarg == Symbol("blocks") && isnothing(blocks) 
                    blocks = arg.args[i+1]
                end
            end
        end
    end
    
    convert_call = Expr(:., :convert, Expr(:tuple, :Int64, threads))

    first_call = Expr(:call, fun_call.args[1], :backend, convert_call, Expr(:call, Expr(:., :KAUtils, QuoteNode(:tuple_mult)), blocks, threads))

    kernel_ass = Expr(Symbol("="), :kernel_call, first_call)

    second_call = Expr(:call, :kernel_call, fun_call.args[2:end]...)

    return Expr(:block, kernel_ass, second_call)
end


#ASSUME CUDA NAMESPACE
function expr_replacer(expr)

    if typeof(expr) != Expr
        return nothing
    end

    if     expr_identify(expr, "(CUDA.threadIdx()).x")
        return Meta.parse("@index(Local, Cartesian)[1]")
    elseif expr_identify(expr, "(CUDA.threadIdx()).y")
        return Meta.parse("@index(Local, Cartesian)[2]")
    elseif expr_identify(expr, "(CUDA.blockIdx()).x")
        return Meta.parse("@index(Group, Cartesian)[1]")
    elseif expr_identify(expr, "(CUDA.blockIdx()).y")
        return Meta.parse("@index(Group, Cartesian)[2]")
    elseif expr_identify(expr, "(CUDA.blockDim()).x")
        return Meta.parse("KernelAbstractions.@groupsize()[1]")
    elseif expr_identify(expr, "(CUDA.blockDim()).y")
        return Meta.parse("KernelAbstractions.@groupsize()[2]")
    elseif expr_identify(expr, "(CUDA.gridDim()).x")
        return Meta.parse("ceil(Int, KernelAbstractions.@ndrange()[1] / KernelAbstractions.@groupsize()[1])")
    elseif expr_identify(expr, "(CUDA.gridDim()).y")
        return Meta.parse("ceil(Int, KernelAbstractions.@ndrange()[2] / KernelAbstractions.@groupsize()[2])")
    elseif expr_identify_1(expr, "CUDA.synchronize")
        new_expr = copy(expr)
        symbol_replace!(new_expr, "CUDA", "KernelAbstractions")
        push!(new_expr.args, Symbol("backend") )
        return new_expr
    elseif expr_identify(expr, "CUDA.sqrt") || expr_identify(expr, "CUDA.abs") || expr_identify(expr, "CUDA.exp") || expr_identify(expr, "CUDA.ceil") || expr_identify(expr, "CUDA.log")  || expr_identify(expr, "CUDA.cos") || expr_identify(expr, "CUDA.atan")
        new_expr = copy(expr)
        symbol_replace!(new_expr, "CUDA", "KernelAbstractions")
        return new_expr

    #ARRAY CURLY CONSTRUCTOR
    elseif expr_identify_1_1(expr, "CUDA.CuArray") && expr.head == :call
        if expr.args[1].head == :curly
            new_expr = copy(expr)
            symbol_replace!(new_expr.args[1], "CUDA", "KAUtils")
            symbol_replace!(new_expr.args[1], "CuArray", "ArrayConstructor")
            insert!(new_expr.args, 2, Symbol("backend"))
            insert!(new_expr.args, 3, extract_type_from_curly_call(expr))
            new_expr.args[1] = uncurlyfy(new_expr.args[1])
            return new_expr
        end
    # ARRAY TYPE
    elseif expr_identify_1(expr, "CUDA.CuArray") && expr.head == :curly
        new_expr = copy(expr)   
        symbol_replace!(new_expr.args[1], "CUDA", "GPUArrays")
        symbol_replace!(new_expr.args[1], "CuArray", "AbstractGPUArray")
        return new_expr
    elseif expr_identify_1(expr, "CUDA.CuDeviceArray") && expr.head == :curly
        new_expr = copy(expr)   
        symbol_replace!(new_expr.args[1], "CUDA", "GPUArrays")
        symbol_replace!(new_expr.args[1], "CuDeviceArray", "AbstractGPUArray")
        return new_expr                       
    #ARRAY CONSTRUCTOR
    elseif expr_identify_1(expr, "CUDA.CuArray") && expr.head == :call
        new_expr = copy(expr)
        symbol_replace!(new_expr, "CUDA", "KAUtils")
        symbol_replace!(new_expr, "CuArray", "ArrayConstructor")
        insert!(new_expr.args, 2, Symbol("backend"))
        return new_expr       

    elseif expr_identify_1(expr, """CUDA.var\"@cuda\"""")
        return generate_kernel_call(expr)

    
    elseif expr_identify_1(expr, """CUDA.var\"@cuprintf\"""")
        return Expr(:macrocall, Symbol("@print"), LineNumberNode(1), expr.args[2:end]...)

    elseif expr_identify_1(expr, """CUDA.var\"@elapsed\"""")
        return Expr(:macrocall, Symbol("@elapsed"), LineNumberNode(1), Expr(:block, expr.args[3], Meta.parse("KernelAbstractions.synchronize(backend)")))


    elseif expr_identify(expr, """\$(Expr(:., :NVTX))""")
        return Expr(:., :KernelAbstractions)
    elseif expr_identify(expr, """\$(Expr(:., :CUDA))""")
        return Expr(:., :KernelAbstractions)

    
    elseif expr_identify(expr, "CUDA.sync_threads()")
        return Expr(:macrocall, Symbol("@synchronize"), LineNumberNode(1)) #TODO add KernelAbstractions namespace


    elseif expr_identify_1(expr, """CUDA.var\"@cuStaticSharedMem\"""")
        T = expr.args[3]
        dims = expr.args[4]
        return Expr(:macrocall, Symbol("@localmem"), LineNumberNode(1), T, dims) #TODO add KernelAbstractions namespace


    elseif expr_identify_1(expr, "CUDA.zeros") || expr_identify_1(expr, "CUDA.ones")
        symbol_replace!(expr, "CUDA", "KernelAbstractions") 
        insert!(expr.args, 2, :backend)
        return expr

    elseif expr_identify_1(expr, "CUDA.copy!")
        symbol_replace!(expr, "CUDA", "KernelAbstractions")
        symbol_replace!(expr, "copy!", "copyto!")  
        insert!(expr.args, 2, :backend)
        return Expr(:block, expr, Meta.parse("KernelAbstractions.synchronize(backend)"))

    ## FORCED VALUES TODO, MAKE THIS PARAMETRIC
    elseif expr_identify(expr, "CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)")
        return Meta.parse("256")
    elseif expr_identify(expr, "CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)") 
        return Meta.parse("1024")
    elseif expr_identify_1(expr, "CUDA.available_memory") 
        return Meta.parse("1024*1024*1024")

    ##
    elseif expr_identify_1(expr, "CUDA.CuDynamicSharedArray")
        emit_warning(DynamicSMArrayWarning())
        emit_warning(DynamicSMArrayToStaticSMArrayWarning())
        return Expr(:macrocall, Symbol("@localmem"), LineNumberNode(1), expr.args[2], expr.args[3])
        
    elseif expr_identify_1(expr, """CUDA.var\"@sync\"""") ## TODO Support for blocking or not blocking sync!
        emit_warning(SyncBlockingForzedWarning())
        return Expr(:block, expr.args[3], Meta.parse("KernelAbstractions.synchronize(backend)"))


    elseif expr_identify_1(expr, "CUDA.ldg")
        return Expr(:ref, expr.args[2], expr.args[3])


    elseif expr_identify_1(expr, """var\"@benchmark\"""")
        kernel_ass = extract_and_empty_kernel_ass!(expr)
        if isnothing(kernel_ass)
            return nothing
        end
        push!(expr.args, Expr(Symbol("="), :setup, kernel_ass))
        return expr
    ## COMMENT VALUES
    elseif expr_identify_line(expr, "CUDA.var\"@profile\"")
        emit_warning(IncompatibleSymbolRemovedWarning("CUDA.@profile"))
        return Meta.parse("""KAUtils.@comment "CUDA.@profile removed by incompatibility"  """ )
    elseif expr_identify_line(expr, "CUDA.device")
        emit_warning(IncompatibleSymbolRemovedWarning("CUDA.device")) 
        return Meta.parse("""KAUtils.@comment "CUDA.device removed by incompatibility"  """ )
    else
        return nothing
    end
end


#Extract type from a curly expression
function extract_type_from_curly_call(expr)
    @assert expr.head == :call
    @assert expr.args[1].head == :curly
    return expr.args[1].args[2]
end


function replace_cuda_2(ast_top, target_backend)

    if target_backend == "CUDA"
        pushfirst!(ast_top.args, Meta.parse("backend = CUDABackend(false, false)"))
        pushfirst!(ast_top.args, Meta.parse("using CUDA"))
    elseif target_backend == "CPU"
        pushfirst!(ast_top.args, Meta.parse("backend = CPU()"))
    elseif target_backend == "AMD"
        pushfirst!(ast_top.args, Meta.parse("backend = ROCBackend()"))
        pushfirst!(ast_top.args, Meta.parse("using AMDGPU"))
    elseif target_backend == "ONEAPI"
        pushfirst!(ast_top.args, Meta.parse("backend = oneAPIBackend()"))
        pushfirst!(ast_top.args, Meta.parse("using oneAPI"))
    elseif target_backend == "METAL"
        pushfirst!(ast_top.args, Meta.parse("backend = MetalBackend()"))
        pushfirst!(ast_top.args, Meta.parse("using Metal"))
    else 
        println("Backend not recognized")
    end

    pushfirst!(ast_top.args, Expr(:using, Expr(:., :., :KAUtils)))
    pushfirst!(ast_top.args, Expr(:call, :include, String(@__DIR__) * "/KAUtils.jl"))
    pushfirst!(ast_top.args, Expr(:using, Expr(:., :GPUArrays)))
    pushfirst!(ast_top.args, Expr(:using, Expr(:., :KernelAbstractions)))

    ast_top = MacroTools.striplines(ast_top)
    str = expr_list_to_string(ast_top.args)
    return str
end




