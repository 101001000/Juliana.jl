include("CUDATranslator.jl")

 
file_input = open(ARGS[1], "r")
file_output = open(ARGS[2], "w")
target_backend = ARGS[3]


str = read(file_input, String)
write(file_output, replace_cuda("begin " * str * " end"))

close(file_input)
close(file_output)
