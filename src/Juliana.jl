module Juliana

	using MacroTools
	using SyntaxTree

	include("warnings.jl")
	include("utils.jl")
	include("preprocessing.jl")
	include("processing.jl")
	include("postprocessing.jl")

	export translate, dump_gpu_info

	function translate(filepath, output_dir)
			
		@debug "Processing " * filepath


		
		ast = load_fat_ast(filepath)

		ast1 = CUDA_symbol_check(ast, true)

		kernel_names = extract_kernelnames(ast1)


		ast2 = expr_replacer(ast1)

		ast3 = attr_replacer(ast2)
		
		ast4 = process_kernels!(ast3, kernel_names)
		

		SyntaxTree.linefilter!(ast4)
		
		ast5 = remove_kernel_annotations(ast4)

		warn_missing_translation(ast5)

		save_fat_ast(ast5, output_dir)
				
		println("Warnings: ")
    	print_warnings()
	end


end
