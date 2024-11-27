import CUDA
import SyntaxTree

function union_dicts!(dict1, dict2)
	for key in keys(dict2)
		new_arr = union(get(dict1, key, Set{Union{Symbol, Expr}}()), dict2[key])
		dict1[key] = new_arr
	end
end

function preprocess(filepaths, extra_knames=[], extra_kfuncs=[])

	asts = []
	deps = Dict()
	kernel_names = Set{Symbol}()
	kernel_funcs = Set{Symbol}()

	for filepath in filepaths
		ast = load_fat_ast(basename(filepath), dirname(filepath))
		ast = CUDA_symbol_check(ast, true)
		ast = remove_unnecessary_prefix(ast)
		ast = wrap_ternary(ast)
		union_dicts!(deps,  extract_dep_graph(ast))
		union!(kernel_names, extract_kernelnames(ast))
		union!(kernel_funcs, extract_kfunctions(ast))
		push!(asts, ast)
	end

	union!(kernel_names, extra_knames)
	union!(kernel_funcs, extra_kfuncs)

	@debug "Kernel names: " * string(kernel_names)
	@debug "Kernel funcs: " * string(kernel_funcs)
	@debug "Function deps: " *string(deps)

	require_ctx_funcs = kernel_funcs

	while true
		old_size = length(require_ctx_funcs)
		for fun in require_ctx_funcs
			union!(require_ctx_funcs, fun in keys(deps) ? deps[fun] : Set{Union{Expr, Symbol}}())
		end
		(old_size != length(require_ctx_funcs)) || break
	end

	#require_ctx_funcs = setdiff(require_ctx_funcs, kernel_names)


	@debug "Require ctx funcs: " * string(require_ctx_funcs)
	return asts, kernel_names, require_ctx_funcs
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
	
	str = comment_encode(str)

	ast = Expr(:file, filepath, Meta.parse("begin " * str * " end").args...)

	ast_fat = MacroTools.postwalk(ast) do node
		if MacroTools.@capture(node, include(includefilepath_))
			if !(includefilepath isa String)
				@warn "Trying to include a file whose path is not known until runtime. Some files may not have been included in the translation"
			else
				sub_ast = load_fat_ast(joinpath(dirname(filepath), includefilepath), ref_dir)
				return sub_ast
			end
		end
		return node
	end

	return ast_fat
end

function comment_encode(str)
	return str
end

# Ternaries don't wrap branches in blocks as :if does.
# This breaks quoting showing, so we force blocks on quote branches in ternaries.
function wrap_ternary(ast)
	ast = MacroTools.postwalk(ast) do node
		if @capture(node, cond_ ? branch1_ : branch2_) # This also captures if.
			change_branches = false
			if branch1 isa Expr
				if branch1.head == :quote
					branch1 = Expr(:block, LineNumberNode(0), branch1)
					change_branches = true
				end
			end
			if branch2 isa Expr
				if branch2.head == :quote
					branch2 = Expr(:block, LineNumberNode(0), branch2)
					change_branches = true
				end
			end
			if change_branches
				return Expr(:if, cond, branch1, branch2)
			end			
		end
		return node
	end
end

processed_defs = Set()

#TODO: Make this module aware.
function extract_dep_graph(ast)
	deps = Dict()
	defs = Set{Union{Expr, Symbol}}()
	caller = nothing
	MacroTools.prewalk(ast) do node

		callerf = capture_fdef_name(node)
		
		if !isnothing(callerf)
			caller = callerf
			try
				if drop_module(callerf) in processed_defs
					emit_warning(DeviceFunctionOverloaded(string(drop_module(callerf))))
				end
				push!(processed_defs, drop_module(callerf))
				push!(defs, drop_module(caller))
			catch
				@error string(drop_module(caller))
			end
		end

		if @capture(node, callee_(fargs__)) && drop_module(callee) != drop_module(caller)
			if isNothing(caller)
				#@error "Function calling without a function caller" * string(node)
			else
				push!(get!(deps, drop_module(callee), Set{Union{Expr, Symbol}}()), drop_module(caller))
			end
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
	knames = Set{Symbol}()
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
	@assert is_fdef(func)
	is_kernel = false
	MacroTools.postwalk(func) do node
		is_kernel |= @capture(node, CUDA.threadIdx()) #TODO: expand this list
		is_kernel |= @capture(node, CUDA.sync_threads())
		return node
	end
	return is_kernel
end

# Retrieve functions which require kernel constructs
function extract_kfunctions(ast)
	kfuncs = Set{Symbol}()
	MacroTools.postwalk(ast) do node
		fname = capture_fdef_name(node)
		if !isnothing(fname)
			if is_kernel_function(node)
				push!(kfuncs, drop_module(fname))
			end
		end
		return node
	end
	return kfuncs
end






# Look for CUDA symbols without the CUDA prefix and add it.
# Store in each scope the symbols overwritten to avoid adding the CUDA namespace to user overwritten symbols.
function CUDA_symbol_check(ast, convert=true)
	cuda_symbols = names(CUDA)
	filter!(symbol -> symbol != :CUDA, cuda_symbols) # remove CUDA to avoid replacing import directives
	envs = [[]]
	ast = skip_prewalk(ast) do node
		if node isa Expr

			# using SOMEPACKAGE: a, b, c... should override CUDA a, b, c
			if node.head == :using
				if node.args[1].head == Symbol(":")
					if node.args[1].args[1] == Expr(:., :CUDA)
						return nothing
					else
						for def in node.args[1].args[2:end]
							push!(envs[end], def.args[1]) 
						end
					end
				else
					return nothing
				end
			end

			# We do not want to override the export symbols in the export clause.
			if node.head == :export 
				return nothing
			end
			
			if @capture(node, struct sname_ sbody__ end)
				for def in sbody
					if def isa Symbol
						push!(envs[end], def)
					end
					if @capture(def, v_::T_)
						push!(envs[end], v)
					end	
				end
			end

			if @capture(node, function name_(args__) body__ end)
				push!(envs[end], name)
				for arg in args
					push!(envs[end], drop_type(arg))
				end
			end
			if @capture(node, name_(args__) = body_)
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
				#@error "Checking" * string(node.args[i]) " is in... " * string(envs)
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

# TODO: There are some cases such CUDA.adapt which in reality is doing Adapt.adapt
# Maybe I could replace such CUDA prefixes with the original ones?
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
							if (isdefined(CUDA, node.args[i+1].value))
								emit_warning(TransitiveCUDAPrefix(string(node)))
								return node
							else
								emit_warning(UnnecessaryCUDAPrefix(string(node)))
								return remove_prefix(node)
							end
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
