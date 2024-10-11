import CUDA
import SyntaxTree

function load_fat_ast(filepath)

	if !isfile(filepath)
		throw(ErrorException(filepath * " not found"))
	end

	file_input = open(filepath, "r")
    str = read(file_input, String)
    close(file_input)
	
	str_ce = comment_encode(str)
	
	ast = Expr(:file, filepath, Meta.parse("begin " * str_ce * " end").args...)

	ast_fat = MacroTools.postwalk(ast) do node
		if MacroTools.@capture(node, include(filename_))
			@debug "File inclussion found, translating recursively..."
			load_path = dirname(filepath) == "" ? filename : dirname(filepath) * "/" * filename
			sub_ast = load_fat_ast(load_path)
			return sub_ast
		end
		return node
	end

	return ast_fat
end

function comment_encode(str)
	return str
end

function extract_kernelnames(ast)
	knames = []
	MacroTools.postwalk(ast) do node
		if @capture(node, CUDA.@cuda args__ kname_(kargs__))
			push!(knames, kname)
		end
		return node
	end
	return knames
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
					push!(envs[end], arg)
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