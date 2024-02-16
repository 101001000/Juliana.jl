import MacroTools
import CUDA

# using MyModule
# in-scope: All exported names (x and y), MyModule.x, MyModule.y, and MyModule.p
# extensible: MyModule.x, MyModule.y, and MyModule.p
# using MyModule: x, p
# in-scope: x and p
# extensible: (nothing)
# import MyModule
# in-scope: MyModule.x, MyModule.y, and MyModule.p
# extensible: MyModule.x, MyModule.y, and MyModule.p
# import MyModule.x, MyModule.p
# in-scope: x and p
# extensible: x and p
# import MyModule: x, p
# in-scope: x and p
# extensible: x and p


# Return a sorted list of the global imports. TODO
#function extract_modules(top_expr)
#    return["CUDA"]
#end
#
#function get_namespace(top_expr, global_symbol)
#    using_modules = extract_modules(top_expr)
#    namespace = nothing
#    for modul in using_modules
#        names = names(modul)
#        for name in names
#            if global_symbol == Symbol(name)
#                namespace = name
#            end
#        end
#    end
#end


# Expression matcher:
# contains_symbol
# exact_string
# 
#


function_call_id = Symbol("") # A terrible global variable which keeps track of the function id.

function extract_kernel_name_from_call(expr)
    for arg in expr.args
        if arg isa Expr
            if arg.head == :call
                return arg.args[1]
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

function remove_farg_types!(expr)
    for i in eachindex(expr.args[1].args)
        if typeof(expr.args[1].args[i]) != Expr 
            continue
        end
        if expr.args[1].args[i].head == Symbol("::")
            expr.args[1].args[i] = expr.args[1].args[i].args[1]
        end
    end
end


function extract_const_vars(expr, s)
    if expr isa Expr
        for arg in expr.args
            if expr_identify_1(arg, "CUDA.ldg")
                push!(s,arg.args[2])
                continue
            end
            extract_const_vars(arg, s)
        end
    end
    return s
end

function constantify_function!(expr, args_ids)
    for i in eachindex(expr.args[1].args[2:end])
        if expr.args[1].args[i] in args_ids
            expr.args[1].args[i] = Expr(:macrocall, Symbol("@Const"), LineNumberNode(1), expr.args[1].args[i])
        end
    end
end

function kernelize_function!(expr, sym)
    for i in eachindex(expr.args)
        if typeof(expr.args[i]) != Expr 
            continue
        end
        if expr.args[i].head == :function
            if(expr.args[i].args[1].args[1] == sym)
                remove_farg_types!(expr.args[i])

                var_ids = Set()

                extract_const_vars(expr.args[i], var_ids)
                constantify_function!(expr.args[i], var_ids)

                expr.args[i] = Expr(:macrocall, Symbol("@kernel"), LineNumberNode(1), expr.args[i])
                continue
            end
        end
        kernelize_function!(expr.args[i], sym)
    end
end

function replace_interpolation(str)
    pattern = r"\$\((Expr\(:\$, :\w+\))\)"
    inside_pattern = r"\$\(Expr\(:\$, :(.*?)\)\)"

    for m in eachmatch(pattern, str)
        new_str = match(inside_pattern, m.match).captures[1]
        str = replace(str, m.match => "\$" * new_str)
    end

    return str
end



function replace_comments(str)
    regex_pattern = r"#.*"
    replace_comment = match -> ";KAUtils.@comment \"\"\"$match\"\"\""
    modified_code = replace(str, regex_pattern => replace_comment; count=typemax(Int))
    return modified_code
end

function undo_replace_comments(str)
    regex_pattern = r"KAUtils\.@comment \"#(.*)\""
    modified_code = replace(str, regex_pattern => s"# \1"; count=typemax(Int))
    return modified_code

end


function expr_to_string(expr)
    io = IOBuffer()
    Base.show_unquoted(io, expr, 0, -1)
    return String(take!(io))
end

function expr_list_to_string(list)
    str = ""
    for el in list
        str *= expr_to_string(el) * "\n"
    end
    return str
end


function expr_identify(expr, str)
    if typeof(expr) != Expr
        return false
    end
    return expr_to_string(expr) == str
end

function expr_identify_1(expr, str)
    try    
        return expr_to_string(expr.args[1]) == str
    catch
        return false
    end
end

function expr_identify_1_1(expr, str)
    try    
        return expr_to_string(expr.args[1].args[1]) == str
    catch
        return false
    end
end

function expr_identify_any(expr, str)
    res = false
    try
        for arg in expr.args
            if expr_to_string(arg) == str
                res = true
            end
            res |= expr_identify_any(arg, str)
        end
    catch
    end
    return res
end

function expr_identify_line(expr, str)
    if typeof(expr) == Expr
        return expr.head != Symbol("block") && expr_identify_any(expr, str) && expr.head != Symbol("function") && expr.head != Symbol("if") 
    end
    return false
end

function expr_identify_n(expr, str)
    try
        for arg in expr.args
            if expr_identify_any(arg, str)
                return true
            end
        end
    catch
    end
    return false
end



function expr_has_symbol(expr, sym)
    if typeof(expr) == Symbol
        return sym == expr
    else
        has_symbol = false
        for arg in expr.args
            has_symbol |= expr_has_symbol(arg, sym)
        end
        return has_symbol
    end
end


function warning_generator(expr)
    if typeof(expr) == Expr
        for arg in expr.args
            if arg == Symbol("CUDA") || arg == Symbol("NVTX") 
                display("UNTRANSLATED SYMBOL")
                display(expr)
            end
            warning_generator(arg)
        end
    end
end

function namespace_replacer(expr)

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
            namespace_replacer(expr.args[i])
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
                end
                if subarg == Symbol("blocks") && isnothing(blocks) 
                    blocks = arg.args[i+1]
                end
            end
        end
    end
    
    convert_call = Expr(:., :convert, Expr(:tuple, :Int64, threads))

    first_call = Expr(:call, fun_call.args[1], :backend, convert_call, Expr(:call, Expr(:., :KAUtils, QuoteNode(:tuple_mult)), blocks, threads))

    second_call = Expr(:call, first_call, fun_call.args[2:end]...)

    return second_call
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


    ## FORCED VALUES TODO, MAKE THIS PARAMETRIC
    elseif expr_identify(expr, "CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)")
        return Meta.parse("256")
    elseif expr_identify(expr, "CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)") 
        return Meta.parse("1024")
    elseif expr_identify_1(expr, "CUDA.available_memory") 
        return Meta.parse("1024*1024*1024")

    ##
    elseif expr_identify_1(expr, "CUDA.CuDynamicSharedArray")
        println("Dynamic Arrays not allowed inside KernelAbstractions kernels. Converted to local static memory. Set the dimensions statically if possible or it wont compile. OFFSET CROPPED")
        return Expr(:macrocall, Symbol("@localmem"), LineNumberNode(1), expr.args[2], expr.args[3])
        
    elseif expr_identify_1(expr, """CUDA.var\"@sync\"""") ## TODO Support for blocking or not blocking sync!
        println("CUDA.@sync forced to be blocking")
        return Expr(:block, expr.args[3], Meta.parse("KernelAbstractions.synchronize(backend)"))


    elseif expr_identify_1(expr, "CUDA.ldg") # TODO make the array constant!
        return Expr(:ref, expr.args[2], expr.args[3])

    ## COMMENT VALUES
    elseif expr_identify_line(expr, "CUDA.var\"@profile\"")  ## TODO, EMIT WARNING
        return Meta.parse("""KAUtils.@comment "Line removed by incompatibility"  """ )
    elseif expr_identify_line(expr, "CUDA.device") 
        return Meta.parse("""KAUtils.@comment "Line removed by incompatibility"  """ )
    else
        return nothing
    end
end



function add_namespace(expr, namespace)
    if typeof(expr) == Symbol
        return Expr(Symbol("."), Symbol(namespace), QuoteNode(expr))
    else
        new_expr = copy(expr)

        if typeof(new_expr.args[1]) == Expr
            new_expr.args[1].args[1] = Expr(Symbol("."), Symbol(namespace), QuoteNode(new_expr.args[1].args[1]))
        else
            new_expr.args[1] = Expr(Symbol("."), Symbol(namespace), QuoteNode(new_expr.args[1]))
        end
        return new_expr
    end
end

function uncurlyfy(expr)
    if typeof(expr) != Expr
        return expr
    end
    if expr.head == Symbol("curly")
        return expr.args[1]
    end
    return expr
end

#Extract type from a curly expression
function extract_type_from_curly_call(expr)
    @assert expr.head == :call
    @assert expr.args[1].head == :curly
    return expr.args[1].args[2]
end



function function_call_is_in_cuda_namespace(expr)
    @assert typeof(expr) == Expr
    @assert expr.head == Symbol("call") || expr.head == Symbol("macrocall") || expr.head == :curly
    n = names(CUDA)
    return uncurlyfy(expr.args[1]) in n
end

function explicit_using_replace!(expr)

    if typeof(expr) != Expr
        return
    end

    for i in eachindex(expr.args)

        arg = expr.args[i]

        if typeof(arg) != Expr
            continue
        end

        if arg.head == Symbol("call") || arg.head == Symbol("macrocall") || arg.head == Symbol("curly")
            if function_call_is_in_cuda_namespace(arg)
                expr.args[i] = add_namespace(arg, "CUDA") 
            end
        end
        explicit_using_replace!(expr.args[i])
    end
end




function replace_cuda_1(ast_top)
    
    namespace_replacer(ast_top)
    warning_generator(ast_top)

    return ast_top
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
    str = replace_interpolation(str)
    #str = undo_replace_comments(str)
end




