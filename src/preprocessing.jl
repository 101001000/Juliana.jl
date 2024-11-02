import CUDA
import SyntaxTree


function preprocess(filepath, extra_knames=[], extra_kfuncs=[])
	ast = load_fat_ast(basename(filepath), dirname(filepath))
	ast = CUDA_symbol_check(ast, true)
	ast = remove_unnecessary_prefix(ast)

	deps = extract_dep_graph(ast)
	kernel_names = extract_kernelnames(ast)	
	kernel_funcs = extract_kfunctions(ast)

	union!(kernel_names, extra_knames)
	union!(kernel_funcs, extra_kfuncs)

	@debug "Kernel names: " * string(kernel_names)
	@debug "Kernel funcs: " * string(kernel_funcs)
	@debug "Function deps: " *string(deps)

	require_ctx_funcs = kernel_funcs

	while true
		old_size = length(require_ctx_funcs)
		for fun in require_ctx_funcs
			union!(require_ctx_funcs, fun in keys(deps) ? deps[fun] : Set())
		end
		(old_size != length(require_ctx_funcs)) || break
	end

	require_ctx_funcs = setdiff(require_ctx_funcs, kernel_names)

	@debug "Require ctx funcs: " * string(require_ctx_funcs)
	return ast, kernel_names, require_ctx_funcs
end

# filename is relative to the main translating file. Dir is where filename is located in a first instance.
function load_fat_ast(filepath, ref_dir)
	
	@info "Loading file in " * joinpath(ref_dir, filepath)

	if !isfile(joinpath(ref_dir, filepath))
		throw(ErrorException(joinpath(ref_dir, filepath) * " not found"))
	end

	file_input = open(joinpath(ref_dir, filepath), "r")
    str = read(file_input, String)
    close(file_input)
	
	str_ce = comment_encode(str)
	
	ast = Expr(:file, filepath, Meta.parse("begin " * str_ce * " end").args...)

	ast_fat = MacroTools.postwalk(ast) do node
		if MacroTools.@capture(node, include(includefilepath_))
			sub_ast = load_fat_ast(joinpath(dirname(filepath), includefilepath), ref_dir)
			return sub_ast
		end
		return node
	end

	return ast_fat
end

function comment_encode(str)
	return str
end

#TODO: Make this module aware.
function extract_dep_graph(ast)
	deps = Dict()
	defs = Set()
	caller = nothing
	MacroTools.prewalk(ast) do node
		if @capture(node, function callerf_(fargs__) body_ end)
			caller = callerf
			push!(defs, drop_module(caller))
		end
		if @capture(node, function callerf_(fargs__) where {T__} body_ end)
			caller = callerf
			push!(defs, drop_module(caller))
		end
		if @capture(node, callee_(fargs__)) && drop_module(callee) != drop_module(caller)
			push!(get!(deps, drop_module(callee), Set()), drop_module(caller))
		end
		return node
	end
	for key in collect(keys(deps))
		if key in defs
			deps[key] = intersect(deps[key], defs)
		else
			delete!(deps, key)
		end
	end
	return deps
end

function uninterpolate(expr)
	if expr isa Expr
		if expr.head == Symbol("\$")
			return expr.args[1]
		end
	end
	return expr
end

function extract_kernelnames(ast)
	knames = []
	MacroTools.postwalk(ast) do node
		if @capture(node, CUDA.@cuda args__ kname_(kargs__))
			push!(knames, uninterpolate(kname))
		end
		return node
	end
	return knames
end

# Check if a function calls kernel constructs.
function is_kernel_function(func)
	@assert(@capture(func, (function fname_(fargs__) fbody_ end) |  (function fname_(fargs__) where {T__} fbody_ end)))
	is_kernel = false
	MacroTools.postwalk(func) do node
		is_kernel |= @capture(node, CUDA.threadIdx()) #TODO: expand this list
		return node
	end
	return is_kernel
end

# Retrieve functions which require kernel constructs
function extract_kfunctions(ast)
	kfuncs = Set()
	MacroTools.postwalk(ast) do node
		if @capture(node, (function fname_(fargs__) fbody_ end) | (function fname_(fargs__) where {T__} fbody_ end))
			#@debug "checking if is kernel " * string(node)
			if is_kernel_function(node)
				push!(kfuncs, drop_module(fname))
			end
		end
		return node
	end
	return kfuncs
end


function push_expr_fun(ast, expr)
	@assert(@capture(ast, function fname_(fargs__) fbody_ end))
	newbody = deepcopy(fbody)
	push!(newbody.args, expr)
	return create_func(fname, fargs, newbody)
end

global f_replacements = Dict{Symbol, Int}()

function replace_returns_fun(ast)
	@assert(@capture(ast, function fname_(fargs__) fbody_ end))
	ocurrences = get!(f_replacements, fname, 0)
	label_name = "end_" * string(fname) * "_" * string(ocurrences)
	var_name = "var_" * string(fname)
	new_ast = MacroTools.prewalk(ast) do node 
		if @capture(node, return retval_)
			return :($(Symbol(var_name)) = $retval; @goto $(Symbol(label_name)))
		end
		return node
	end
	f_replacements[fname] = ocurrences + 1
	return push_expr_fun(new_ast, :(@label $(Symbol(label_name))))
end





# Look for CUDA symbols without the CUDA prefix and add it.
# Store in each scope the symbols overwritten to avoid adding the CUDA namespace to user overwritten symbols.
function CUDA_symbol_check(ast, convert=true)
	cuda_symbols = names(CUDA)
	filter!(symbol -> symbol != :CUDA, cuda_symbols) # remove CUDA to avoid replacing import directives
	envs = [[]]
	ast = MacroTools.prewalk(ast) do node
		if node isa Expr
			if @capture(node, function name_(args__) body__ end)
				push!(envs[end], name)
				for arg in args
					push!(envs[end], drop_type(arg))
				end
			end
			if @capture(node, function name_(args__) where {types__} body__ end)
				for type in types
					push!(envs[end], type)
				end
			end
			if @capture(node, name_ = x_)
				if name isa Expr # in case it is a tuple
					append!(envs[end], name.args)
				else
					push!(envs[end], name)
				end
			end
			if node.head == :block
				push!(envs, [])
			end
			for i in eachindex(node.args)
				if node.args[i] in cuda_symbols && !(node.args[i] in (vcat(envs...)))
					if convert
						node.args[i] = wrap_with_namespace(node.args[i])
						return node
					end
				end
			end
			if node.head == :block
				pop!(envs)
			end
		end
		return node
	end
	return ast
end

function remove_prefix(expr)
	@assert(expr.head == Symbol("."))
	return expr.args[2].value
end

function remove_unnecessary_prefix(ast)
	cuda_symbols = names(CUDA, all=true)
	new_ast = skip_prewalk(ast) do node
		if node isa Expr
			if node.head == :using 
				return nothing
			end
			for i in eachindex(node.args) 
				if node.args[i] == :CUDA
					try
						if !(node.args[i+1].value in cuda_symbols)
							emit_warning(UnnecessaryCUDAPrefix(string(node)))
							return remove_prefix(node)
						end
					catch
						@error string(node) * " case not considered for namespace"
					end
				end
			end
		end
		return node
	end
	return new_ast
end
