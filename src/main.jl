include("CUDATranslator.jl")

kernel_ids = Set()
asts = []

for file_name in ARGS[1:end-2]

    println("Loading ", file_name)

    file_input = open(file_name, "r")
    str = read(file_input, String)
    str = "begin " * str * " end"
    close(file_input)

    #str = replace_comments(str)
    ast = Meta.parse(str)

    explicit_using_replace!(ast)
    extract_kernel_names!(ast, kernel_ids)
    push!(asts, ast)
end

for id in kernel_ids
    for ast in asts
        kernelize_function!(ast, id)
    end
end


for i in eachindex(ARGS[1:end-2])

    file_input = ARGS[i]

    println("Outputing ", ARGS[end-1] * basename(file_input))


    ast = replace_cuda_1(asts[i])
    str = replace_cuda_2(ast, ARGS[end])
    #str = string(i)
    file_output = open(ARGS[end-1] * basename(file_input), "w")
    write(file_output, str)
    close(file_output)

end
