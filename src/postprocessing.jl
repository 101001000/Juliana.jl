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
	mkpath(dirname(output_dir * "/" * thin_ast.args[1]))
	@info "Storing file in " * output_dir * "/" * thin_ast.args[1] 
	file_output = open(output_dir * "/" * ast.args[1], "w")
    write(file_output, node_to_string(thin_ast))
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

function remove_linenumber_nodes(ast)
	rem_ast = MacroTools.postwalk(ast) do node
		if node isa Expr
			return Expr(node.head, filter(arg -> !(arg isa LineNumberNode), node.args)...)
		else
			return node
		end
	end
	return rem_ast
end

function warn_missing_translation(ast)
	ast = skip_prewalk(ast) do node
		if node isa Expr
			for arg in node.args
				if arg == :CUDA
					emit_warning(UntranslatedWarning(string(node)))
				end
			end
		end
		return node
	end
	return ast
end