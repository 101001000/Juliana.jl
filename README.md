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
Juliana.translate_file(
    "path/to/file.jl",
    "path/to/output"
)

# Translate multiple files
Juliana.translate_files(
    ["path/to/file1.jl", "path/to/file2.jl"],
    ["path/to/output1", "path/to/output2"]
)
# Note that for multiple files, the number of output paths must be equal to the number of input files

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
Juliana don't work well with GPU-CPU function overload (e.g writing a function with the same name which has different body depending if runs on the GPU or the CPU). It will attemp to fix some of this cases by itself, but it has some limitations. A clear separation of the GPU and CPU functions is recommended.

Juliana also has some issues when overriding existing CUDA functions (e.g. creating your own `device` function). Try to avoid clashing CUDA symbols with your own ones.


## Feature support
| Feature | Support Status |
| ------- | ------------- |
| CUDA Kernels | Full support |
| Shared Memory | Full support (static only) |
| Device Functions | Full support |
| Synchronization | Full support |
| Thread Indexing | Full support |
| Memory Management | Full support |
| Stream Operations | Partial support |
| Dynamic Parallelism | Not supported |
| Texture Memory | Not supported |
| Warp Operations | Not supported |


## Warning list
| Code | Type | Description | Impact |
|------|------|-------------|---------|
| WN001 | UntranslatedWarning | Untranslated CUDA symbol | Non-critical |
| WN002 | SyncBlockingForzedWarning | CUDA @sync forced to be blocking | Performance impact |
| WN003 | DynamicSMArrayWarning | Dynamic Shared Memory Arrays not allowed inside KernelAbstractions kernels | Breaking |
| WN004 | DynamicSMArrayToStaticSMArrayWarning | Dynamic Shared Memory Array converted to Static Shared Memory Array | Requires const size |
| WN005 | IncompatibleSymbolRemovedWarning | CUDA Symbol removed by incompatibility | Breaking |
| WN006 | ThreadSizeNotChecked | Thread size not checked for max size | Performance risk |
| WN007 | ThreadSizeTooLarge | Thread size shouldn't exceed max size for compatibility | Compatibility risk |
| WN008 | DeviceAttributeWarning | Device attributes are loaded from a config file emulating Nvidia GPU | Simulation only |
| WN009 | ImplicitCudaNamespace | Implicit namespace candidate symbol found | Code conflict risk |
| WN010 | UnsupportedKWArg | Keyword argument in kernel call not supported | Breaking |
| WN011 | AttributeSimulated | A hardcoded attribute has been used to replace a CUDA attribute | Simulation only |
| WN012 | FreeMemorySimulated | Free memory simulated with 4GB default for some backends | Resource limitation |
| WN013 | UnnecessaryCUDAPrefix | Code used wrongly a CUDA namespace prefix | Code style |
| WN014 | UnprocessedKernels | Some kernels were not found | Breaking |
| WN015 | TransitiveCUDAPrefix | Code used a CUDA namespace prefix for accessing other module | Code style |
| WN016 | NoConstMemory | CUDA Const device array specifier removed | Performance impact |
| WN017 | DeviceFunctionOverloaded | GPU function with context propagation used in CPU | Potential conflict |

## Translated projects
[Julia Rodinia](https://github.com/JuliaParallel/rodinia) benchmarks. Full translation, no changes required.

[Julia MiniBUDE](https://github.com/UoB-HPC/miniBUDE/tree/main/src/julia/miniBUDE.jl). Full translation, minimal changes required (`device` function name clash).

[Julia BabelStream](https://github.com/UoB-HPC/BabelStream/tree/main/src/julia/JuliaStream.jl). Full translation, minimal changes required (`device` function name clash).

[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl). Full translation, no changes required. Only AMD/NVIDIA (FFTW requires unified memory).

## Citation
Proceeding of the congress where the project was presented are not released yet.
