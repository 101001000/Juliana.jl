#TODO: using CUDA splitting.

function postprocess(asts, output_dirs)
	#SyntaxTree.linefilter!(ast) #this breaks quoting
	i = 1
	for ast in asts
		ast = append_usingKA(ast)
		ast = remove_kernel_annotations(ast)
		ast = merge_interp_symbol(ast)
		ast = MacroTools.flatten(ast)
		warn_missing_translation(ast)
		save_fat_ast(ast, output_dirs[i])
		i = i + 1
	end
end

function append_usingKA(ast)
	ast = MacroTools.postwalk(ast) do node
		if node isa Expr
			if node.head == :using
				for arg in node.args
					if arg == Expr(:., :CUDA) # using A, CUDA, B... TODO: nuke CUDA import
						return Expr(:block, Expr(:using, node.args..., Expr(:., :Juliana), Expr(:., :GPUArrays)), Expr(:import, Expr(:., :KernelAbstractions)))
					end
					if arg.head == Symbol(":") # using CUDA: ...
						if arg.args[1] == Expr(:., :CUDA)
							return Expr(:block, Expr(:using, Expr(:., :CUDA), Expr(:., :Juliana), Expr(:., :GPUArrays)), Expr(:import, Expr(:., :KernelAbstractions)))
						end
					end
				end
			end
		end
		return node
	end
	return ast
end

function save_fat_ast(ast, output_dir)
	@assert ast isa Expr
	@assert ast.head == :file || ast.head == :hidden_file
	thin_ast = MacroTools.prewalk(ast) do node
		if node isa Expr
			if (node.head == :file || node.head == :hidden_file) && node != ast
				if node.args[1] != ast.args[1]
					filepath = node.args[1]
					save_fat_ast(node, output_dir)
					filepath = dirname(ast.args[1]) == "" ? filepath : relpath(filepath, dirname(ast.args[1]))
					if node.head == :file
						return :(include($(filepath)))
					else
						return nothing
					end
				end
			end
		end
		return node
	end
	output_path = joinpath(output_dir, thin_ast.args[1])
	mkpath(dirname(output_path))
	@info "Storing file in " * output_dir * "/" * thin_ast.args[1] 
	file_output = open(output_dir * "/" * ast.args[1], "w")
	str = node_to_string(thin_ast)
	str = replace_quoting_block(str)  
	str = replace_quoting(str) 
	str = replace_var_interpolation(str)
	str = replace_dot_interpolation(str)

    write(file_output, str)
    close(file_output)
end

# Apply the same transformation multiple times
function apply_transformation(str, f)
	max_it = 10
	new_str = ""
	i = 0
	while str != new_str
		if i >= max_it
			@error "String transformation possibly recursive"
		end
		new_str = f(str)
		i = i + 1
	end

	return new_str
end

# Fuse $symbol expressions
function merge_interp_symbol(ast)
	ast = MacroTools.postwalk(ast) do node
		if node isa Expr
			if node.head == Symbol(raw"$") && length(node.args) == 1
				if node.args[1] isa Symbol
					return Symbol(raw"$"*String(node.args[1]))
				end
			end
		end
		return node
	end

	return ast
end

function show_unquoted(io::IO, ex::Expr, indent::Int, prec::Int, quote_level::Int = 0)
	if ex.head == :file
		for ex in ex.args[2:end]
			print(io, '\n', " "^indent)
			Base.show_unquoted(io, ex, indent, -1, quote_level)
		end
	else
		Base.show_unquoted(io, ex, indent, prec, quote_level)
	end
end

function remove_kernel_annotations(ast)
	rem_ast = MacroTools.prewalk(ast) do node
		if node isa Expr
			if node.head == :kernel
				return node.args[1]
			end
		end
		return node
	end
	return rem_ast
end

function warn_missing_translation(ast)
	cuda_symbols = names(CUDA, all=true)
	ast = skip_prewalk(ast) do node
		if node isa Expr
			if node.head == :using 
				return nothing
			end
			for i in eachindex(node.args) 
				if node.args[i] == :CUDA
					try
						if node.args[i+1].value in cuda_symbols
							emit_warning(UntranslatedWarning(string(node)))
						end
					catch
					end
				end
			end
		end
		return node
	end
	return ast
end


#function split_seq(str)
#	pattern = Regex(raw"(?s)\$\(Expr\(:seq, (.*?)\)\)")
#    inside_pattern = Regex(raw"(?s)\$\(Expr\(:seq, (.*?)\)\)")
#    for m in eachmatch(pattern, str)
#        new_str = match(inside_pattern, m.match).captures[1]
#        str = replace(str, m.match => "quote " * new_str * " end")
#    end
#    return str
#end
