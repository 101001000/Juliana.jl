module Juliana

	using MacroTools
	using SyntaxTree
	using TOML

	include("warnings.jl")
	include("utils.jl")
	include("KAUtils.jl")
	include("preprocessing.jl")
	include("processing.jl")
	include("postprocessing.jl")

	using .KAUtils

	export translate_file, translate_pkg, dump_gpu_info
	export KAUtils

	function translate_pkg(pkg_input_path, pkg_output_path)
		pkg_name = TOML.parsefile(pkg_input_path * "/Project.toml")["name"]
		@info "Translating the package " * pkg_name
		main_file_path = pkg_input_path * "/src/" * pkg_name * ".jl"
		Base.Filesystem.cptree(pkg_input_path, pkg_output_path, force=true)
		translate_file(main_file_path, pkg_output_path * "/src/")		
	end

	function translate_file(filepath, output_dir)
		@info "Translating " * filepath
		ast, kernel_names = preprocess(filepath)
		ast = process(ast, kernel_names)
		ast = postprocess(ast, output_dir)	
		println("Warnings: ")
    	print_warnings()
	end


end
