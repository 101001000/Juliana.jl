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
["CUDA.CuArray", "AbstractGPUArray"],

["CUDA.synchronize()", "KernelAbstractions.synchronize(backend)"],
["CUDA.device()", "nothing", IncompatibleSymbolRemovedWarning("CUDA.device()")]
]


# This only works with function calls.
function unsplat_fargs(ast)
    new_ast = prewalk_once(ast) do node
	    if @capture(node, f_(arg_...))
			return Expr(node.head, f, arg...)
		end
        return node
    end
    return new_ast
end


function constantify_kernel(ast)
	const_args = []	
	ast1 = MacroTools.postwalk(ast) do node
		if @capture(node, CUDA.ldg(var_, i_))
			push!(const_args, var)
			return Expr(:ref, var, i)
		end
		return node
	end
	ast2 = MacroTools.postwalk(ast1) do node
		if @capture(node, @kernel function fname_(fargs__) fbody_ end)
			new_args = []
			for arg in fargs
				if arg in const_args
					push!(new_args, Expr(:macrocall, Symbol("@Const"), LineNumberNode(1, :none), arg))
				else
					push!(new_args, arg)
				end
			end
			return macro_wrap(create_func(fname, new_args, fbody), Symbol("@kernel"))
		end
		return node
	end 
	return ast2
end

function macro_wrap(ast, macro_symbol)
	return Expr(:macrocall, macro_symbol, LineNumberNode(1, :none), ast)
end

# Quoting makes this anoying double body thing. 
function create_func(name, args, body)
	f = :(function $name($args...) $body end)
	f.args[2].args = f.args[2].args[3].args
	return unsplat_fargs(f)
end


function process_kernel(ast)
	ast = replace_returns_fun(ast)
	ast = macro_wrap(ast, Symbol("@kernel"))
	ast = constantify_kernel(ast)
	return ast
end

function process_kernels(ast, kernel_names)
	new_ast = MacroTools.postwalk(ast) do node
		if @capture(node, function fname_(fargs__) fbody_ end)
			if fname in kernel_names
				return process_kernel(node)
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
				if checkbounds(Bool, replacement, 3)
					emit_warning(replacement[3])
				end
                return unsplat_fargs(merged_ast)
            end
        end
        return node
    end
    return new_ast
end
