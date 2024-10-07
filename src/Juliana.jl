module Juliana

	using MacroTools

	include("warnings.jl")
	include("utils.jl")
	include("preprocessing.jl")
	include("processing.jl")
	include("postprocessing.jl")

	export translate, dump_gpu_info

	function translate(filepath, output_dir)
			
		@debug "Processing " * filepath
		
		ast = load_fat_ast(filepath)

		ast = CUDA_symbol_check(ast, true)

		ast = expr_replacer(ast)

		ast = attr_replacer(ast)

		ast_rem = remove_linenumber_nodes(ast)

		warn_missing_translation(ast)

		save_fat_ast(ast, output_dir)
				
		println("Warnings: ")
    	print_warnings()
	end


end
