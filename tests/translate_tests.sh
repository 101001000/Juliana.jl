backend=CUDA

rm rodinia/julia_gen/ -d -r -f

cp rodinia/julia_cuda/ rodinia/julia_gen/ -d -r
find rodinia/julia_gen/ -name "*.jl" -type f -delete

julia -e 'using Juliana; main("--input rodinia/julia_cuda --output rodinia/julia_gen --recursive --backend='$backend'")'

rm ExaTronKernels.jl/test/GEN.jl -f
julia -e 'using Juliana; main("--input ExaTronKernels.jl/test/CUDA.jl --output ExaTronKernels.jl/test/GEN.jl --recursive --backend='$backend'")'

