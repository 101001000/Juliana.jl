import CUDA
import SyntaxTree


function preprocess(filepath)
	ast = load_fat_ast(basename(filepath), dirname(filepath))
	ast = CUDA_symbol_check(ast, true)
	ast = remove_unnecessary_prefix(ast)
	kernel_names = extract_kernelnames(ast)
	deps, defs = extract_dep_graph(ast)
	fnames_to_inline = setdiff(extract_fnames_to_inline(ast, deps), kernel_names)
	ast = fcall_inliner(ast, defs, fnames_to_inline)
	return ast, kernel_names
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

function extract_dep_graph(ast)
	defs = Dict() # A map of fname -> fast
	deps = Dict()
	caller = nothing
	MacroTools.prewalk(ast) do node
		if @capture(node, function fname_(fargs__) body_ end)
			defs[fname] = node
			caller = fname		
		end
		if @capture(node, fname_(fargs__)) && fname != caller
			push!(get!(deps, caller, []), fname)
		end
		return node
	end
	for key in keys(deps)
		deps[key] = intersect(deps[key], keys(defs))
	end
	return deps, defs
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

function letify_func(ast, args_map)
	@assert(@capture(ast, function fname_(fargs__) fbody_ end))
	var = Symbol("var_" * string(fname))
	ast = replace_returns_fun(ast)
	new_ast = MacroTools.postwalk(ast) do node
		if node in keys(args_map)
			return args_map[node]
		else
			return node
		end
	end
	Expr(:let, Expr(:block), Expr(:block, new_ast.args[2], var))
end


function extract_fnames_to_inline(ast, deps)
	return []
end


#TODO: This inlining has issues with arguments replacement.
#	function test(a)
#		a *= 2
#		return a + 1
#	end
#
#	test(1) -> let 
#					1*=2
#					1+1 instead 2 + 1
#				end

function fcall_inliner(ast, fmap, fnames_to_inline)
    new_ast = prewalk_parent_info((node, parent) -> begin
        if @capture(node, fname_(fargs__)) && !@capture(parent, function fname2_(fargs2__) fbody2_ end)
            if fname in fnames_to_inline 
				args_map = Dict()
				for i in eachindex(fargs)
					args_map[fmap[fname].args[1].args[i+1]] = fargs[i] 
				end
                return letify_func(fmap[fname], args_map)
            end
        end
        return node
    end, ast, nothing)
    return new_ast
end

function drop_type(expr)
	return expr #TODO
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
