# JULIANA (**J**ulia **U**nification **L**ayer for **I**ntel, **A**MD, **N**vidia and **A**pple)

Translation from CUDA -> KernelAbstraction is the only translation supported right now.


Sample usage: 
```julia console
> using(Julana); main("--input fileinput1.jl --outpupt fileoutput.jl --backend=(CUDA|ONEAPI|METAL|CPU|AMD) --recursive")
```

Working [Julia Rodinia](https://github.com/JuliaParallel/rodinia/tree/master/julia_cuda) benchmarks:
- [x] backprop
- [x] bfs
- [x] hotspot
- [x] leukocyte
- [x] lud 
- [x] nn
- [x] nw
- [x] particlefilter_double 
- [x] pathfinder
- [x] streamcluster


Working [ExaTronKernels](https://github.com/exanauts/ExaTronKernels.jl) benchmarks:

It's necessary to remove the comment from CUDA.jl at the beginning of the file quoted with """ as juliana doesn´t support multiline comments yet, and It's also necessary to mark the n variable as const, and remove n as an argument for all functions as KernelAbstractions doesn´t allow dynamic shared memory allocation. All this changes are reflected in my [ExaTronKernels fork](https://github.com/101001000/ExaTronKernels.jl)

- [x] dicf 
- [x] dicfs 
- [x] dcauchy
- [x] dtrpcg0
- [x] dprsrch
- [x] daxpy 
- [x] dssyax
- [x] dmid
- [x] dgpstep
- [x] dbreakpt
- [x] dnrm2
- [x] nrm2
- [x] dcopy
- [x] ddot
- [x] dscal
- [x] dtrqsol
- [x] dspcg
- [x] dgpnorm
- [x] dtron
- [ ] driver_kernel -> requires some changes in how functions called from the inside of kernels are handled.

