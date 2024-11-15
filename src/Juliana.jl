module Juliana

	using MacroTools
	using SyntaxTree
	using TOML

	include("warnings.jl")
	include("utils.jl")
	include("quoting_handling.jl")
	include("KAUtils.jl")
	include("preprocessing.jl")
	include("processing.jl")
	include("postprocessing.jl")

	using .KAUtils

	export translate_file, translate_pkg, dump_gpu_info
	export KAUtils

	function translate_pkg(pkg_input_path, pkg_output_path, extra_files=[], extra_knames=[], extra_kfuncs=[], gpu_sim="NVIDIA_GeForce_GTX_950")
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
		run_tests_file_path = pkg_input_path * "/test/runtests.jl"
		Base.Filesystem.cptree(pkg_input_path, pkg_output_path, force=true)

		if isfile(run_tests_file_path)
			push!(extra_files, run_tests_file_path)
		end

		translate_files(vcat(main_file_path, extra_files), [pkg_output_path * "/src/", pkg_output_path * "/test/"], extra_knames, extra_kfuncs, gpu_sim)		
	end

	function translate_files(filepaths, output_dirs, extra_knames=[], extra_kfuncs=[], gpu_sim="NVIDIA_GeForce_GTX_950")
		@info "Translating " * string(filepaths)
		asts, kernel_names, require_ctx_funcs = preprocess(filepaths, extra_knames, extra_kfuncs)
		asts = process(asts, kernel_names, require_ctx_funcs, gpu_sim)
		asts = postprocess(asts, output_dirs)	
		println("Warnings: ")
    	print_warnings()
	end


end
