# JULIANA (**J**ulia **U**nification **L**ayer for **I**ntel, **A**MD, **N**vidia and **A**pple)

Juliana is a syntax translation tool for [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package to [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). It will translate a big portion of CUDA.jl functions and macros to KernelAbstractions.jl equivalent constructs.


## Installation and usage:


### Installation
```bash
git clone https://github.com/101001000/Juliana.jl
julia -e 'using Pkg; Pkg.develop(path="./Juliana.jl")'
```


### Usage
```bash
using Juliana

# Translate single file
Juliana.translate_files(
    ["path/to/file.jl"],
    ["path/to/output1"]
)

# Translate multiple files
Juliana.translate_files(
    ["path/to/file1.jl", "path/to/file2.jl"],
    ["path/to/output1", "path/to/output2"]
)

# Translate an entire package
Juliana.translate_pkg(
    "path/to/package",
    "path/to/output-package"
)
```

### Additional arguments:
- `extra_knames`: Additional list of kernel names to be included.
- `extra_kfuncs`: Additional list of kernel functions to be included in the output.
- `gpu_sim`: GPU simulator to be used. Default: "NVIDIA_GeForce_GTX_950"

Use this arguments when Juliana is not able to find the proper definitions of the kernels or functions.

### Extra considerations:



## Feature support
<details>
<br>
- [x] Indexing
- [x] Ldg to @Const
</details>



## Translated projects
[Julia Rodinia](https://github.com/JuliaParallel/rodinia) benchmarks. Full translation, no changes required.

[Julia MiniBUDE](https://github.com/UoB-HPC/miniBUDE/tree/main/src/julia/miniBUDE.jl). Full translation, minimal changes required (`device` function name clash).

[Julia BabelStream](https://github.com/UoB-HPC/BabelStream/tree/main/src/julia/JuliaStream.jl). Full translation, minimal changes required (`device` function name clash).

[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl). Full translation, no changes required. Only AMD/NVIDIA (FFTW requires unified memory).

## Citation
Proceeding of the congress where the project was presented are not released yet.
