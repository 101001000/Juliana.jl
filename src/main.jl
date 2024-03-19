using ArgParse
using FilePathsBase

include("codeprocessor.jl")
include("namespaceresolver.jl")
include("exprreplacer.jl")
include("kernelizer.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--input", "-i"
            help = "input files separated by a comma, or directory"
            required = true
        "--output", "-o"
            help = "output files separated by a comma, or directory"
            required = true
        "--recursive", "-r"
            help = "travel directories recursively"
            action = :store_true
        "--prefix", "-p"
            help = "prefix for translated files"
        "--backend", "-b"
            help = "output backend desired"
            default = "CUDA"
        "--comments", "-c"
            help = "enable comments"
            action = :store_true
        "--mirror", "-m"
            help = "mirror the input file structure in the output"
            action = :store_true
        "--inliner-depth"
            help = "the amount of inlining depth desired. Set it to 0 to avoid inlining completely or -1 to inline completely (be careful with recursive functions)"
            default = -1
            arg_type = Int
    end
    return parse_args(s)
end

function get_files_from_dir(path, recursive)
    files = []
    str_list = readdir(path)
    str_list = map(x -> path * "/" * x, str_list)
    for file in str_list
        if isfile(file) && splitext(file)[2] == ".jl"
            push!(files, file)
        elseif isdir(file)
            if recursive
                append!(files, get_files_from_dir(file, recursive))
            end
        else
            #println("ehhhmmmm $file")
        end
    end
    return files
end

function retrieve_input_files(files_str_list, recursive)
    files = []
    for path in files_str_list
        if isfile(path)
            push!(files, path)
        elseif isdir(path)
            append!(files, get_files_from_dir(path, recursive))
        else
            println("$path does not exist")
        end
    end
    return files
end

function retrieve_files(parsed_args)

    input_paths  = split(parsed_args["input"], ",")
    output_paths = split(parsed_args["output"], ",")

    if isdir(input_paths[1]) && !isfile(output_paths[1]) && length(input_paths) == 1 && length(output_paths) == 1
        println("dir mode")
        input_files  = retrieve_input_files(input_paths, parsed_args["recursive"])
        output_files = map(x -> replace(x, input_paths[1] => output_paths[1]), input_files)
    elseif length(input_paths) == length(output_paths) && all(path -> isfile(path), input_paths) && all(path -> !isdir(path), output_paths)
        println("file mode")  
        input_files = input_paths
        output_files = output_paths
    else
        println("wrong usage")
        for path in input_paths
            if !isfile(path)
                println(path, " not found")
            end
        end
    end

    if !isnothing(parsed_args["prefix"])
        output_files = map(x -> dirname(x) * "/" * parsed_args["prefix"] * basename(x), output_files)
    end

    return input_files, output_files
end


function main()

    parsed_args = parse_commandline()
    input_files, output_files = retrieve_files(parsed_args)

    kernel_ids = Set()
    fs = Set()
    asts = []
    deps = Dict()

    for file_name in input_files

        println("Loading ", file_name)
    
        file_input = open(file_name, "r")
        str = read(file_input, String)
        str = "begin " * str * " end"
        close(file_input)
    
        if parsed_args["comments"]
            str = replace_comments(str)
        end
        
        ast = Meta.parse(str)
    
        explicit_using_replace!(ast)
        extract_kernel_names!(ast, kernel_ids) # Maybe is better to extract directly the kernel AST?

        extract_functions!(ast, fs, deps)          
        namespace_replacer!(ast)
        namespace_replacer!(ast) # need to run this twice because the postprocessing step required for @benchmark.
        push!(asts, ast)
    end

    fs_names = map(x -> x.args[1].args[1], collect(fs))
    for dep in values(deps)
        filter!(x -> x in fs_names, dep)
    end
    unroll_deps!(deps, first(keys(deps)), nothing)

    is_kernel_dict = Dict(f.args[1].args[1] => false for f in fs)
    process_is_kernel!(fs, deps, is_kernel_dict)

    for i in eachindex(output_files)
        while true
            ast_pre_block_cleanup = copy(asts[i])
            block_cleaner!(asts[i])
            ast_pre_block_cleanup != asts[i] || break
        end
    
        warning_generator(asts[i])
     
        # fs_inlined is a dictionary where the keys are the functions to be inlined, and the value, the amount of replacements.
        fs_inlined = Dict(f => 0 for f in filter(x -> is_kernel_dict[x.args[1].args[1]], fs))

        for id in kernel_ids
            kernelize_function!(asts[i], id, fs_inlined, parsed_args["inliner-depth"])
        end 

        str = replace_cuda_2(asts[i], parsed_args["backend"])
        str = replace_interpolation(str)

        if parsed_args["comments"]
            str = undo_replace_comments(str)
        end
        
        mkpath(dirname(output_files[i]))
        file_output = open(output_files[i], "w")
        write(file_output, str)
        close(file_output)
    end


end

main()


