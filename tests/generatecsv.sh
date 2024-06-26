julia save_benchmarks.jl "rodinia.csv" "rodinia/julia_gen/backprop/bpnn_layerforward_CUDA.json,backprop,gen;
rodinia/julia_cuda/backprop/bpnn_layerforward_CUDA.json,backprop,cuda;
rodinia/julia_gen/backprop/bpnn_adjust_weights_cuda.json,backprop,gen;
rodinia/julia_cuda/backprop/bpnn_adjust_weights_cuda.json,backprop,cuda;
rodinia/julia_gen/bfs/Kernel.json,bfs,gen;
rodinia/julia_cuda/bfs/Kernel.json,bfs,cuda;
rodinia/julia_gen/bfs/Kernel2.json,bfs,gen;
rodinia/julia_cuda/bfs/Kernel2.json,bfs,cuda;
rodinia/julia_gen/hotspot/calculate_temp.json,hotspot,gen;
rodinia/julia_cuda/hotspot/calculate_temp.json,hotspot,cuda;
rodinia/julia_gen/leukocyte/GICOV_kernel.json,leukocyte,gen;
rodinia/julia_cuda/leukocyte/GICOV_kernel.json,leukocyte,cuda;
rodinia/julia_gen/leukocyte/dilate_kernel.json,leukocyte,gen;
rodinia/julia_cuda/leukocyte/dilate_kernel.json,leukocyte,cuda;
rodinia/julia_gen/leukocyte/IMGVF_kernel.json,leukocyte,gen;
rodinia/julia_cuda/leukocyte/IMGVF_kernel.json,leukocyte,cuda;
rodinia/julia_gen/lud/lud_diagonal.json,lud,gen;
rodinia/julia_cuda/lud/lud_diagonal.json,lud,cuda;
rodinia/julia_gen/lud/lud_perimeter.json,lud,gen;
rodinia/julia_cuda/lud/lud_perimeter.json,lud,cuda;
rodinia/julia_gen/lud/lud_internal.json,lud,gen;
rodinia/julia_cuda/lud/lud_internal.json,lud,cuda;
rodinia/julia_gen/nn/euclid.json,nn,gen;
rodinia/julia_cuda/nn/euclid.json,nn,cuda;
rodinia/julia_gen/nw/needle_cuda_shared_1.json,nw,gen;
rodinia/julia_cuda/nw/needle_cuda_shared_1.json,nw,cuda;
rodinia/julia_gen/nw/needle_cuda_shared_2.json,nw,gen;
rodinia/julia_cuda/nw/needle_cuda_shared_2.json,nw,cuda;
rodinia/julia_gen/particlefilter/likelihood_kernel.json,particlefilter,gen;
rodinia/julia_cuda/particlefilter/likelihood_kernel.json,particlefilter,cuda;
rodinia/julia_gen/particlefilter/sum_kernel.json,particlefilter,gen;
rodinia/julia_cuda/particlefilter/sum_kernel.json,particlefilter,cuda;
rodinia/julia_gen/particlefilter/normalize_weights_kernel.json,particlefilter,gen;
rodinia/julia_cuda/particlefilter/normalize_weights_kernel.json,particlefilter,cuda;
rodinia/julia_gen/particlefilter/find_index_kernel.json,particlefilter,gen;
rodinia/julia_cuda/particlefilter/find_index_kernel.json,particlefilter,cuda;
rodinia/julia_gen/pathfinder/dynproc_kernel.json,pathfinder,gen;
rodinia/julia_cuda/pathfinder/dynproc_kernel.json,pathfinder,cuda;
rodinia/julia_gen/streamcluster/kernel_compute_cost.json,streamcluster,gen;
rodinia/julia_cuda/streamcluster/kernel_compute_cost.json,streamcluster,cuda"

julia save_aggregated_benchmarks.jl "rodinia-aggregated-gen.csv" "rodinia/julia_gen/backprop/backprop-aggregated.json;
rodinia/julia_gen/bfs/bfs-aggregated.json;
rodinia/julia_gen/hotspot/hotspot-aggregated.json;
rodinia/julia_gen/leukocyte/leukocyte-aggregated.json;
rodinia/julia_gen/lud/lud-aggregated.json;
rodinia/julia_gen/nn/nn-aggregated.json;
rodinia/julia_gen/nw/nw-aggregated.json;
rodinia/julia_gen/particlefilter/particlefilter-aggregated.json;
rodinia/julia_gen/pathfinder/pathfinder-aggregated.json;
rodinia/julia_gen/streamcluster/streamcluster-aggregated.json"

julia save_aggregated_benchmarks.jl "rodinia-aggregated-cuda.csv" "rodinia/julia_cuda/backprop/backprop-aggregated.json;
rodinia/julia_cuda/bfs/bfs-aggregated.json;
rodinia/julia_cuda/hotspot/hotspot-aggregated.json;
rodinia/julia_cuda/leukocyte/leukocyte-aggregated.json;
rodinia/julia_cuda/lud/lud-aggregated.json;
rodinia/julia_cuda/nn/nn-aggregated.json;
rodinia/julia_cuda/nw/nw-aggregated.json;
rodinia/julia_cuda/particlefilter/particlefilter-aggregated.json;
rodinia/julia_cuda/pathfinder/pathfinder-aggregated.json;
rodinia/julia_cuda/streamcluster/streamcluster-aggregated.json"

julia normalize_csvs.jl "rodinia-aggregated-cuda.csv" "rodinia-aggregated-gen.csv" 

julia save_benchmarks.jl "exatronkernels.csv" "ExaTronKernels.jl/results/gen/dicf.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dicf.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dicfs.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dicfs.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dcauchy.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dcauchy.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dtrpcg.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dtrpcg.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dprsrch.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dprsrch.json,exatron,cuda;
ExaTronKernels.jl/results/gen/daxpy.json,exatron,gen;
ExaTronKernels.jl/results/cuda/daxpy.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dssyax.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dssyax.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dmid.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dmid.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dgpstep.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dgpstep.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dbreakpt.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dbreakpt.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dnrm2.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dnrm2.json,exatron,cuda;
ExaTronKernels.jl/results/gen/nrm2.json,exatron,gen;
ExaTronKernels.jl/results/cuda/nrm2.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dcopy.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dcopy.json,exatron,cuda;
ExaTronKernels.jl/results/gen/ddot.json,exatron,gen;
ExaTronKernels.jl/results/cuda/ddot.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dscal.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dscal.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dtrqsol.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dtrqsol.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dspcg.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dspcg.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dgpnorm.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dgpnorm.json,exatron,cuda;
ExaTronKernels.jl/results/gen/dtron.json,exatron,gen;
ExaTronKernels.jl/results/cuda/dtron.json,exatron,cuda"


