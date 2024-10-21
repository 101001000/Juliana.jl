
function postprocess(ast, output_dir)
	SyntaxTree.linefilter!(ast)
	ast = remove_kernel_annotations(ast)
	warn_missing_translation(ast)
	save_fat_ast(ast, output_dir)
end

function save_fat_ast(ast, output_dir)
	@assert ast isa Expr
	@assert ast.head == :file
	thin_ast = MacroTools.postwalk(ast) do node
		if node isa Expr
			if node.head == :file && node != ast
				if node.args[1] != ast.args[1]
					filepath = node.args[1]
					save_fat_ast(node, output_dir)
					return :(include($(filepath)))
				end
			end
		end
		return node
	end
	output_path = joinpath(output_dir, thin_ast.args[1])
	@info dirname(output_path)
	mkpath(dirname(output_path))
	@info "Storing file in " * output_dir * "/" * thin_ast.args[1] 
	file_output = open(output_dir * "/" * ast.args[1], "w")
	str = node_to_string(thin_ast)
	str = replace_interpolation(str)
    write(file_output, str)
    close(file_output)
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
					if node.args[i+1].value in cuda_symbols
						emit_warning(UntranslatedWarning(string(node)))
					end
				end
			end
		end
		return node
	end
	return ast
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