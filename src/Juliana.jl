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

		kernel_names = extract_kernelnames(ast)
		ast, kernel_asts = identify_kernels(ast, kernel_names)


		@info "kernel names: " * string(kernel_names)


		ast = expr_replacer(ast)

		ast = attr_replacer(ast)
		
		for kernel_ast in kernel_asts
			kernel_ast.args[1] = kernel_wrap(kernel_ast.args[1])
			kernel_ast.args[1] = constantify_kernel(kernel_ast.args[1])
		end
		
		println(kernel_asts)
		

		ast_rem = remove_linenumber_nodes(ast)
		
		ast_rem = remove_kernel_annotations(ast_rem)

		warn_missing_translation(ast_rem)

		save_fat_ast(ast_rem, output_dir)
				
		println("Warnings: ")
    	print_warnings()
	end


end
