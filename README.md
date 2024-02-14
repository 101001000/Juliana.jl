# JULIANA (**J**ulia **U**nification **L**ayer for **I**ntel, **A**MD, **N**vidia and **A**pple)


Only single files and a very limited feature set is supported right now.

Translation from CUDA -> KernelAbstraction is the only translation supported right now.


Usage: 
```console
~$ julia fileinput.jl fileoutput.jl (CUDA|ONEAPI|METAL|CPU|AMD)
```

Working [Julia Rodinia](https://github.com/JuliaParallel/rodinia/tree/master/julia_cuda) benchmarks:
- [ ] backprop -> requires multifile
- [x] bfs
- [x] hotspot
- [ ] leukocyte -> requires multifile
- [ ] lud -> requires multifile
- [x] nn
- [ ] nw -> requires multifile
- [x] particlefilter_double
- [x] pathfinder
- [ ] streamcluster -> requires multifile


Working [ExaTronKernels](https://github.com/exanauts/ExaTronKernels.jl) benchmarks:

It's necessary to remove the comment from CUDA.jl at the beginning of the file quoted with """ as juliana doesn´t support multiline comments yet, and It's also necessary to mark the n variable as const, and remove n as an argument for all functions as KernelAbstractions doesn´t allow dynamic shared memory allocation. All this changes are reflected in my [ExaTronKernels fork](https://github.com/exanauts/ExaTronKernels.jl)

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

