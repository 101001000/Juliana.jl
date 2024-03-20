using CUDA
using JSON


config_parameters = Dict()

function add_parameter(name, value)
    config_parameters[name] = value
end

function load_950_data()
    open(joinpath(@__DIR__ ,  "gpu-info/NVIDIA_GeForce_GTX_950.json"), "r") do file
        data = JSON.parse(file)
        for att_key in keys(data)
            add_parameter(string(att_key), data[att_key])
        end
    end
end

# Auxiliary function which creates a file with the data of the current GPU
function dump_gpu_info()
    dev = device()
    atts = Dict()
    for CUatt in instances(CUDA.CUdevice_attribute_enum)
        display(CUatt)
        try
            atts[CUatt] = attribute(dev, CUatt)
        catch
            println("not available")
        end
    end
    jsonString = JSON.json(atts)
    open("gpu-info/" * replace(name(dev), " " => "_") * ".json", "w") do file
        write(file, jsonString)
    end
end

load_950_data()

add_parameter("max_threadsize_warning",                      256)
add_parameter("available_memory",                            1024*1024*1024)
