julia -e "using Pkg; Pkg.develop(path=\"../\")"

echo "Translating rodinia..."

rm -rf rodinia-KA
cp rodinia rodinia-KA -r

julia -e "using Juliana; translate_file(\"rodinia/julia_cuda/backprop/facetrain.jl\", \"rodinia-KA\")"
julia -e "using Juliana; translate_file(\"rodinia/julia_cuda/bfs/bfs.jl\", \"rodinia-KA\")"
julia -e "using Juliana; translate_file(\"rodinia/julia_cuda/hotspot/hotpot.jl\", \"rodinia-KA\")"
julia -e "using Juliana; translate_file(\"rodinia/julia_cuda/leukocyte/leukocyte.jl\", \"rodinia-KA\")"
julia -e "using Juliana; translate_file(\"rodinia/julia_cuda/lud/lud.jl\", \"rodinia-KA\")"
julia -e "using Juliana; translate_file(\"rodinia/julia_cuda/nn/nn.jl\", \"rodinia-KA\")"
julia -e "using Juliana; translate_file(\"rodinia/julia_cuda/nw/needle.jl\", \"rodinia-KA\")"
julia -e "using Juliana; translate_file(\"rodinia/julia_cuda/particlefilter/particlefilter_double.jl\", \"rodinia-KA\")"
julia -e "using Juliana; translate_file(\"rodinia/julia_cuda/pathfinder/pathfinder.jl\", \"rodinia-KA\")"
julia -e "using Juliana; translate_file(\"rodinia/julia_cuda/streamcluster/streamcluster.jl\", \"rodinia-KA\")"


echo "Translating ExaTron..."

rm -rf ExaTronKernels.jl-KA

julia -e "using Juliana; translate_pkg(\"ExaTronKernels.jl\", \"ExaTronKernels.jl-KA\")"
