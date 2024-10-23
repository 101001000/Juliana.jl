echo Testing Backprop
(cd rodinia-KA/julia_cuda/backprop/; ./verify)
echo Testing BFS
(cd rrodinia-KA/julia_cuda/bfs/; ./verify)
echo Testing Hotspot
(cd rodinia-KA/julia_cuda/hotspot/; ./verify)
echo Testing Leukocyte
(cd rodinia-KA/julia_cuda/leukocyte/; ./verify)
echo Testing LUD
(cd rodinia-KA/julia_cuda/lud/; ./verify)
echo Testing NN
(cd rodinia-KA/julia_cuda/nn/; ./verify)
echo Testing NW
(cd rodinia-KA/julia_cuda/nw/; ./verify)
echo Testing ParticleFilter
(cd rodinia-KA/julia_cuda/particlefilter/; ./verify)
echo Testing PathFinder
(cd rodinia-KA/julia_cuda/pathfinder/; ./verify)
echo Testing StreamCluster
(cd rodinia-KA/julia_cuda/streamcluster/; ./verify)
echo Testing ExaTronKernels
(cd ExaTronKernels.jl-KA/test/; julia CUDA.jl)
