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

	function translate_pkg(pkg_input_path, pkg_output_path, extra_knames=[], extra_kfuncs=[])
		toml_path = pkg_input_path * "/Project.toml"
		toml_file = TOML.parsefile(toml_path)
		toml_file["deps"]["KernelAbstractions"] = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
		toml_file["deps"]["Juliana"] = "2e90fba2-f937-4406-aa62-769b40e98753"
		toml_file["deps"]["GPUArrays"] = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
		open(toml_path, "w") do io
			TOML.print(io, toml_file)
		end
		pkg_name = toml_file["name"]
		@info "Translating the package " * pkg_name
		main_file_path = pkg_input_path * "/src/" * pkg_name * ".jl"
		Base.Filesystem.cptree(pkg_input_path, pkg_output_path, force=true)
		translate_file(main_file_path, pkg_output_path * "/src/", extra_knames, extra_kfuncs)		
	end

	function translate_file(filepath, output_dir, extra_knames=[], extra_kfuncs=[])
		@info "Translating " * filepath
		ast, kernel_names, require_ctx_funcs = preprocess(filepath, extra_knames, extra_kfuncs)
		ast = process(ast, kernel_names, require_ctx_funcs)
		ast = postprocess(ast, output_dir)	
		println("Warnings: ")
    	print_warnings()
	end


end
