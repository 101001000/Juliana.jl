backend=ONEAPI

rm rodinia/julia_gen/ -d -r -f
cp rodinia/julia_cuda/ rodinia/julia_gen/ -d -r
find rodinia/julia_gen/ -name "*.jl" -type f -delete

julia ../src/main.jl rodinia/julia_cuda/backprop/backprop.jl rodinia/julia_cuda/backprop/backprop_cuda.jl rodinia/julia_cuda/backprop/backprop_cuda_kernel.jl rodinia/julia_cuda/backprop/facetrain.jl rodinia/julia_gen/backprop/ $backend
julia ../src/main.jl rodinia/julia_cuda/bfs/bfs.jl rodinia/julia_gen/bfs/ $backend
julia ../src/main.jl rodinia/julia_cuda/hotspot/hotspot.jl rodinia/julia_gen/hotspot/ $backend
julia ../src/main.jl rodinia/julia_cuda/leukocyte/find_ellipse.jl rodinia/julia_cuda/leukocyte/find_ellipse_kernel.jl rodinia/julia_cuda/leukocyte/leukocyte.jl rodinia/julia_cuda/leukocyte/misc_math.jl rodinia/julia_cuda/leukocyte/track_ellipse.jl rodinia/julia_cuda/leukocyte/track_ellipse_kernel.jl rodinia/julia_gen/leukocyte/ $backend
julia ../src/main.jl rodinia/julia_cuda/lud/common.jl rodinia/julia_cuda/lud/lud.jl rodinia/julia_cuda/lud/lud_kernel.jl rodinia/julia_gen/lud/ $backend
julia ../src/main.jl rodinia/julia_cuda/nn/nn.jl rodinia/julia_gen/nn/ $backend
julia ../src/main.jl rodinia/julia_cuda/nw/needle.jl rodinia/julia_cuda/nw/needle_kernel.jl rodinia/julia_gen/nw/ $backend
julia ../src/main.jl rodinia/julia_cuda/particlefilter/particlefilter_double.jl rodinia/julia_gen/particlefilter/ $backend
julia ../src/main.jl rodinia/julia_cuda/pathfinder/pathfinder.jl rodinia/julia_gen/pathfinder/ $backend
julia ../src/main.jl rodinia/julia_cuda/streamcluster/streamcluster.jl rodinia/julia_cuda/streamcluster/streamcluster_cuda.jl rodinia/julia_cuda/streamcluster/streamcluster_header.jl rodinia/julia_gen/streamcluster/ $backend
