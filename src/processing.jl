using JSON

replacements = [
# Indexing replacements: 
["CUDA.threadIdx().x", "KernelAbstractions.@index(Local, Cartesian)[1]"],
["CUDA.threadIdx().y", "KernelAbstractions.@index(Local, Cartesian)[2]"],
["CUDA.threadIdx().z", "KernelAbstractions.@index(Local, Cartesian)[3]"],

["CUDA.blockIdx().x", "KernelAbstractions.@index(Group, Cartesian)[1]"],
["CUDA.blockIdx().y", "KernelAbstractions.@index(Group, Cartesian)[2]"],
["CUDA.blockIdx().z", "KernelAbstractions.@index(Group, Cartesian)[3]"],

["CUDA.blockDim().x", "KernelAbstractions.@groupsize()[1]"],
["CUDA.blockDim().y", "KernelAbstractions.@groupsize()[2]"],
["CUDA.blockDim().z", "KernelAbstractions.@groupsize()[3]"],

["CUDA.gridDim().x", "ceil(Int, KernelAbstractions.@ndrange()[1] / KernelAbstractions.@groupsize()[1])"],
["CUDA.gridDim().y", "ceil(Int, KernelAbstractions.@ndrange()[2] / KernelAbstractions.@groupsize()[2])"],
["CUDA.gridDim().z", "ceil(Int, KernelAbstractions.@ndrange()[3] / KernelAbstractions.@groupsize()[3])"],




#CuArray constructor:
["CUDA.CuArray(args__)", "KAUtils.ArrayConstructor(KAUtils.get_backend(), args...)"],
["CUDA.CuArray{t_}(args__)", "KAUtils.ArrayConstructor(KAUtils.get_backend(), t, args...)"],
["CUDA.CuArray{t_, d_}(args__)", "KAUtils.ArrayConstructor(KAUtils.get_backend(), t, args...)"], #TODO: I'm ignoring here dimensions. Check if this can be done

#CuDeviceArray constructor:
["CUDA.CuDeviceArray(args__)", "GPUArrays.AbstractGPUArray(args...)"],
["CUDA.CuDeviceArray{t1_}(args__)", "GPUArrays.AbstractGPUArray{t1}(args...)"],
["CUDA.CuDeviceArray{t1_, t2_}(args__)", "GPUArrays.AbstractGPUArray{t1, t2}(args...)"],
["CUDA.CuDeviceArray{t1_, t2_, t3_}(args__)", "GPUArrays.AbstractGPUArray{t1, t2, t3}(args...)"],

#CuArray typ:

["v_::CUDA.CuArray", "v::GPUArrays.AbstractGPUArray"],
["v_::CUDA.CuArray{t__}", "v::GPUArrays.AbstractGPUArray{t...}"],
["v_::CUDA.CuDeviceArray", "v::KAUtils.DeviceArray"],
["v_::CUDA.CuDeviceArray{t1_}", "v::KAUtils.DeviceArray{t1}"],
["v_::CUDA.CuDeviceArray{t1_, t2_}", "v::KAUtils.DeviceArray{t1, t2}"],
["v_::CUDA.CuDeviceArray{t1_, t2_, t3_}", "v::KAUtils.DeviceArray{t1, t2, t3}"],


["CUDA.CuArray", "GPUArrays.AbstractGPUArray"],
["CUDA.CuArray{t__}", "GPUArrays.AbstractGPUArray{t...}"],
["CUDA.CuDeviceArray", "KAUtils.DeviceArray"],
["CUDA.CuDeviceArray{t__}", "DenKAUtils.DeviceArrayseArray{t...}"],


["CUDA.CuDynamicSharedArray(T_, dims_)", "KernelAbstractions.@localmem T dims", DynamicSMArrayToStaticSMArrayWarning()],
["CUDA.CuDynamicSharedArray(T_, dims_, off_)", "KernelAbstractions.@localmem T dims", DynamicSMArrayToStaticSMArrayWarning()],
["CUDA.@cuStaticSharedMem(T_, dims_)", "KernelAbstractions.@localmem T dims"],

["CUDA.synchronize()", "KernelAbstractions.synchronize(KAUtils.get_backend())"],
["CUDA.sync_threads()", "KernelAbstractions.@synchronize()"],
["CUDA.@sync(body_)", "begin body ; KernelAbstractions.synchronize(KAUtils.get_backend()) end"],

["CUDA.@cuprintln(args__)", "KernelAbstractions.@print(args...)"], #TODO: add line jump


#["CUDA.device()", "nothing", IncompatibleSymbolRemovedWarning("CUDA.device()")],
#["CUDA.@profile discard__", "nothing", IncompatibleSymbolRemovedWarning("CUDA.@profile")],
#["CUDA.WMMA.x_", "nothing", IncompatibleSymbolRemovedWarning("CUDA.WMMA")],

# CUDA Address Aliases
["CUDA.AS.Generic", "0"],
["CUDA.AS.Global", "1"],
["CUDA.AS.Shared", "3"],
["CUDA.AS.Constant", "4"],
["CUDA.AS.Local", "5"],


["CUDA.has_cuda(args__)", "true"],
["CUDA.has_cuda_gpu(args__)", "true"],
["CUDA.CUDABackend(args__)", "KAUtils.get_backend()"],
["CUDA.ndevices()", "1"],
["CUDA.device!(args__)", "nothing"],
["CUDA.CuDevice(args__)", "KAUtils.Device(args...)"],
["CUDA.CuDevice", "KAUtils.Device"],
["CUDA.device()", "KAUtils.device()"],
["CUDA.devices()", "KAUtils.devices()"],
["CUDA.name(args__)", "KAUtils.name(args...)"],


["CUDA.zeros(args__)", "KAUtils.zeros(KAUtils.get_backend(), args...)"],
["CUDA.ones(args__)", "KAUtils.ones(KAUtils.get_backend(), args...)"],
["CUDA.fill(args__)", "KAUtils.fill(KAUtils.get_backend(), args...)"],

["CUDA.@atomic exp_", "KernelAbstractions.@atomic exp"],


["CUDA.available_memory()", "KAUtils.available_memory(KAUtils.get_backend())", FreeMemorySimulated()],
["CUDA.default_rng()", "KAUtils.default_rng(KAUtils.get_backend())"],

#["using CUDA", "using CUDA, KernelAbstractions, Juliana, GPUArrays"],

]

function process(asts, kernel_names, require_ctx_funcs, gpu_sim)
	processed_asts = []
	processed_kernels = []

	for ast in asts
		ast = add_ctx(ast, require_ctx_funcs, kernel_names)
		ast = expr_replacer(ast)
		ast = attr_replacer(ast, gpu_sim)
		ast = kcall_replacer(ast)
		ast = process_kernels!(ast, kernel_names, processed_kernels)
		push!(processed_asts, ast)
	end

	pending_kernels = setdiff(kernel_names, processed_kernels)
	if !isempty(pending_kernels)
		emit_warning(UnprocessedKernels(string(pending_kernels)))
	end

	return processed_asts
end


function add_ctx(ast, require_ctx_funcs, kernel_names)
	ast = MacroTools.postwalk(ast) do node
		fname = capture_fdef_name(node)
		if !isnothing(fname) && drop_module(fname) in require_ctx_funcs && drop_module(fname) != :__init__
			return add_ctx_calls(node, require_ctx_funcs, kernel_names)
		end
		return node
	end

	ast = MacroTools.postwalk(ast) do node
		if @capture(node, fname_(fargs__))
			if drop_module(fname) in require_ctx_funcs && drop_module(fname) != :__init__  && !(fname in kernel_names)

				if :__ctx__ in node.args
					return node
				else 
					new_node = deepcopy(node)
					insert!(new_node.args, head_is(get(node.args, 2, nothing), :parameters) ? 3 : 2, :(nothing))
					return new_node
				end
			end
		end
		return node
	end

	return ast
end

function add_ctx_calls(ast, require_ctx_funcs, kernel_names)
	ast = MacroTools.postwalk(ast) do node
		if @capture(node, fname_(fargs__))
			if drop_module(fname) in require_ctx_funcs && drop_module(fname) != :__init__  && !(fname in kernel_names)
				new_node = deepcopy(node)
				insert!(new_node.args, head_is(get(node.args, 2, nothing), :parameters) ? 3 : 2, :__ctx__)	
				return new_node
			end
		end
		return node
	end
	return ast
end



function macro_wrap(ast, macro_symbol)
	return Expr(:macrocall, macro_symbol.args[1], LineNumberNode(1, :none), ast)
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
		if @capture(node, KernelAbstractions.@kernel function fname_(fargs__) fbody_ end)
			new_args = []
			for arg in fargs
				if arg in const_args
					push!(new_args, Expr(:macrocall, :(KernelAbstractions.@Const), LineNumberNode(1, :none), arg))
				else
					push!(new_args, arg)
				end
			end
			return macro_wrap(create_func(fname, new_args, fbody), :(KernelAbstractions.@kernel))
		end
		return node
	end 
	return ast2
end


#TODO: I'm nuking the genereics in kernels. Maybe this should be revisited...
function drop_generics_fdef(ast)
	if @capture(ast, function fname_(fargs__) where {ftypes__} body_ end)
		return create_func(fname, fargs, body)
	end
	return ast
end


#TODO: This doesn´t work with Ts...
function push_expr_fun(func, expr)
	@assert(is_fdef(func))
	fname, fargs, fbody, _ = capture_fdef(func)
	newbody = Expr(:block, deepcopy(fbody))
	push!(newbody.args, expr)
	return create_func(fname, fargs, newbody)
end

global f_replacements = Dict{Symbol, Int}()

function replace_returns_fun(func)
	fname = capture_fdef_name(func)
	@assert(!isnothing(fname), "trying to replace returns of a non function " * string(func))
	ocurrences = get!(f_replacements, fname, 0)
	label_name = "end_" * string(fname) * "_" * string(ocurrences)
	var_name = "var_" * string(fname)
	new_ast = MacroTools.prewalk(func) do node 
		if @capture(node, return retval_)
			return :($(Symbol(var_name)) = $retval; @goto $(Symbol(label_name)))
		end
		return node
	end
	f_replacements[fname] = ocurrences + 1
	return push_expr_fun(new_ast, :(@label $(Symbol(label_name))))
end



function process_kernel(ast)
	ast = replace_returns_fun(ast)
	ast = macro_wrap(ast, :(KernelAbstractions.@kernel))
	ast = constantify_kernel(ast)
	return ast
end


function process_kernels!(ast, kernel_names, processed_kernels)
	new_ast = MacroTools.postwalk(ast) do node
		fname, fargs, fbody, _ = capture_fdef(node)
		if is_fdef(node)
			if fname in kernel_names
				push!(processed_kernels, fname)
				return process_kernel(node)
			end
		end
		return node
	end
	return new_ast
end


global kcalls_replaced = 0

function kcall_replacer(ast)
	new_ast = MacroTools.prewalk(ast) do node
		if @capture(node, CUDA.@cuda kwargs__ kname_(kargs__))
			kwargs_dict = Dict()
			kwargs_dict[:threads] = 1 # Default thread value
			kwargs_dict[:blocks] = 1 # TODO: I'm not sure about this, or if blocks are calculated automatically. Maybe not use nd_range?
			supported_kwargs = [:blocks, :threads]
			#all available kwargs: :dynamic, :launch, :kernel, :name, :always_inline, :minthreads, :maxthreads,
			#:blocks_per_sm, :maxregs, :fastmath, :cap, :ptx, :cooperative, :blocks, :threads, :shmem, :stream
			for kwarg in kwargs
				@assert(@capture(kwarg, kwname_ = kwval_)) #TODO: By now we just process assignments. Maybe is required to extend this later? https://github.com/JuliaGPU/CUDA.jl/blob/a8fecea4b730c153dd79c8245dc1dc71fe8e095b/src/compiler/execution.jl#L31
				kwargs_dict[kwname] = kwval
				if !(kwname in supported_kwargs)
					emit_warning(UnsupportedKWArg(string(kwname)))
				end
			end
			convert_call = Expr(:., :convert, Expr(:tuple, :Int64, kwargs_dict[:threads]))
			tuple_mult = Expr(:call, Expr(:., :KAUtils, QuoteNode(:tuple_mult)), kwargs_dict[:blocks], kwargs_dict[:threads])
    		
			kcall_name = Symbol("kernel_call_" * string(kcalls_replaced))

			kernel_ass = Expr(Symbol("="), kcall_name, Expr(:call, kname, :(KAUtils.get_backend()), convert_call, tuple_mult))
			kernel_call = Expr(:call, kcall_name, kargs...)
		
			global kcalls_replaced += 1;
			return Expr(:block, kernel_ass, kernel_call)
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

function attr_replacer(ast, gpu)
   attr_dict = Dict() 
   open(joinpath(@__DIR__ ,  "../gpu-presets/$(gpu).json"), "r") do file
        data = JSON.parse(file)
        for attr_key in keys(data)
            attr_dict[string(attr_key)] = data[attr_key]
        end
    end
    new_ast = MacroTools.postwalk(ast) do node
        if @capture(node, CUDA.attribute(dev_, attr_)) 
			emit_warning(AttributeSimulated(string(attr.args[2].value)))
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
                return unsplat_fcallargs(merged_ast)
            end
        end
        return node
    end
    return new_ast
end
