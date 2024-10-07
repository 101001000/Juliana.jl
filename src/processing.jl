using JSON

replacements = [
# Indexing replacements: 
["CUDA.threadIdx().x", "KernelAbstractions.@index(Local, Cartesian)[1]"],
["CUDA.threadIdx().y", "KernelAbstractions.@index(Local, Cartesian)[2]"],
["CUDA.threadIdx().z", "KernelAbstractions.@index(Local, Cartesian)[3]"],

["CUDA.blockIdx().x", "@index(Group, Cartesian)[1]"],
["CUDA.blockIdx().y", "@index(Group, Cartesian)[2]"],
["CUDA.blockIdx().z", "@index(Group, Cartesian)[3]"],

["CUDA.blockDim().x", "KernelAbstractions.@groupsize()[1]"],
["CUDA.blockDim().y", "KernelAbstractions.@groupsize()[2]"],
["CUDA.blockDim().z", "KernelAbstractions.@groupsize()[3]"],

["CUDA.gridDim().x", "ceil(Int, KernelAbstractions.@ndrange()[1] / KernelAbstractions.@groupsize()[1])"],
["CUDA.gridDim().y", "ceil(Int, KernelAbstractions.@ndrange()[2] / KernelAbstractions.@groupsize()[2])"],
["CUDA.gridDim().z", "ceil(Int, KernelAbstractions.@ndrange()[3] / KernelAbstractions.@groupsize()[3])"],

["CUDA.CuArray(args__)", "KAUtils.ArrayConstructor(args...)"],
["CUDA.CuArray{t_}(args__)", "KAUtils.ArrayConstructor{t}(args...)"],

["v_::CUDA.CuArray", "v::AbstractGPUArray"],
["CUDA.CuArray", "AbstractGPUArray"]



]

# This only works with function calls.
function unsplatter(ast)
    new_ast = MacroTools.postwalk(ast) do node
	    if node isa Expr
            if @capture(node, f_(arg_...))
                return Expr(node.head, f, arg...)
            end
        end
        return node
    end
    return new_ast
end

function merge_env(ast, env)
    new_ast = MacroTools.postwalk(ast) do node
	    return node isa Symbol && haskey(env, node) ? env[node] : node
    end
    return new_ast
end

function attr_replacer(ast, gpu="NVIDIA_GeForce_GTX_950")
   attr_dict = Dict() 
   open(joinpath(@__DIR__ ,  "../gpu-presets/$(gpu).json"), "r") do file
        data = JSON.parse(file)
        for attr_key in keys(data)
            attr_dict[string(attr_key)] = data[attr_key]
        end
    end
    new_ast = MacroTools.postwalk(ast) do node
        if @capture(node, CUDA.attribute(dev_, attr_)) 
	        return Meta.parse(string(attr_dict["CU_" * string(attr.args[2].value)]))
	    end
        return node
    end
    return new_ast    
end

function expr_replacer(ast)
    new_ast = skip_prewalk(ast) do node
        for replacement in replacements
            env = MacroTools.trymatch(Meta.parse(replacement[1]), node)
            if !isnothing(env)
                merged_ast = merge_env(Meta.parse(replacement[2]), env)
                return unsplatter(merged_ast)
            end
        end
        return node
    end
    return new_ast
end
