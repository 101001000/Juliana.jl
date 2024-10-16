module Juliana

	using MacroTools
	using SyntaxTree

	include("warnings.jl")
	include("utils.jl")
	include("KAUtils.jl")
	include("preprocessing.jl")
	include("processing.jl")
	include("postprocessing.jl")

	using .KAUtils

	export translate, dump_gpu_info

	function translate(filepath, output_dir)
			
		@debug "Processing " * filepath


		
		ast = load_fat_ast(filepath)

		ast = CUDA_symbol_check(ast, true)

		kernel_names = extract_kernelnames(ast)

		deps, defs = extract_dep_graph(ast)

		fnames_to_inline = setdiff(extract_fnames_to_inline(ast, deps), kernel_names)

		ast = fcall_inliner(ast, defs, fnames_to_inline)

		@info "deps: " * string(deps)



		ast = expr_replacer(ast)

		ast = attr_replacer(ast)

		ast = kcall_replacer(ast)
		
		@info "Kernels found: " * string(kernel_names)

		ast = process_kernels(ast, kernel_names)
		




		SyntaxTree.linefilter!(ast)
		
		ast = remove_kernel_annotations(ast)

		warn_missing_translation(ast)

		save_fat_ast(ast, output_dir)
				
		println("Warnings: ")
    	print_warnings()
	end


end
