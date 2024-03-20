struct Warning
    warningcode::String
    warningname::String
    message::String
    data::String
end

const warning_list = [
    Warning("WN001", "UntranslatedWarning", "Untranslated CUDA symbol", "empty"),
    Warning("WN002", "SyncBlockingForzedWarning", "CUDA @sync forced to be blocking", ""),
    Warning("WN003", "DynamicSMArrayWarning", "Dynamic Shared Memory Arrays not allowed inside KernelAbstractions kernels", ""),
    Warning("WN004", "DynamicSMArrayToStaticSMArrayWarning", "Dynamic Shared Memory Array converted to Static Shared Memory Array. Offset cropped, please ensure the size is marked as const", "")
]

for warning in warning_list
    func_name = Symbol(warning.warningname)
    warning_code = warning.warningcode
    warning_message = warning.message
    
    if warning.data == "" 
        @eval begin
            $func_name() = Warning($warning_code, $(string(func_name)), $warning_message, "")
        end
    else  # For warnings that require additional data
        @eval begin
            $func_name(data::String) = Warning($warning_code, $(string(func_name)), $warning_message, data)
        end
    end
end

emitted_warnings = Warning[]

function emit_warning(warning::Warning)
    push!(emitted_warnings, warning)
end

function print_warnings()
    dict = Dict()

    for warning in emitted_warnings
        if haskey(dict, warning)
            dict[warning] += 1
        else
            dict[warning] = 1
        end
    end

    display(dict)
end