#TODO: encapsulate this into a Warning manager.
struct Warning
    warningcode::String
    warningname::String
    message::String
    data::String
end

#TODO: change the empty thing for additional data.
warning_list = [
    Warning("WN001", "UntranslatedWarning", "Untranslated CUDA symbol", "empty"),
    Warning("WN002", "SyncBlockingForzedWarning", "CUDA @sync forced to be blocking", ""),
    Warning("WN003", "DynamicSMArrayWarning", "Dynamic Shared Memory Arrays not allowed inside KernelAbstractions kernels", ""),
    Warning("WN004", "DynamicSMArrayToStaticSMArrayWarning", "Dynamic Shared Memory Array converted to Static Shared Memory Array. Offset cropped, please ensure the size is marked as const", ""),
    Warning("WN005", "IncompatibleSymbolRemovedWarning", "CUDA Symbol removed by incompatibility", "empty"),
    Warning("WN006", "ThreadSizeNotChecked", "Thread size not checked for max size", ""),
    Warning("WN007", "ThreadSizeTooLarge", "Thread size shouldn't exceed max size for compatibility with older devices", ""),
    Warning("WN008", "DeviceAttributeWarning", "Device attributes are loaded from a config file emulating some Nvidia GPU", ""),
    Warning("WN009", "ImplicitCudaNamespace", "Implicit namespace candidate symbol found, make sure your code and imports are not overriding CUDA symbols.", "empty"),
    Warning("WN010", "UnsupportedKWArg", "Keyword argument in kernel call not supported.", "empty"),
    Warning("WN011", "AttributeSimulated", "A hardcoded attribute has been used to replace a CUDA attribute", "empty"),
    Warning("WN012", "FreeMemorySimulated", "Free memory could not be available in some backends, a default value of 4GB is used in those cases.", ""),
	Warning("WN013", "UnnecessaryCUDAPrefix", "Code used wrongly a CUDA namespace prefix", "empty") #TODO: this can be solved in the preprocessing step.
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

function print_warnings_dict(warnings_dict)
    # Function to wrap text and ensure proper alignment
    function wrap_text(text, max_length)
        if length(text) > max_length
            return text[1:max_length-3] * "..."
        else
            return text
        end
    end

    # Aggregate warnings with wrapped descriptions
    aggregated_warnings = Dict()
    for (warning, count) in warnings_dict
        key = (warning.warningcode, warning.warningname, wrap_text(warning.message, 32))
        if haskey(aggregated_warnings, key)
            push!(aggregated_warnings[key], (warning.data, count))
        else
            aggregated_warnings[key] = [(warning.data, count)]
        end
    end

	if isempty(aggregated_warnings)
		return
	end

    # Determine dynamic padding based on the longest entry for each field
    max_warning_code_length = maximum([length(key[1]) for key in keys(aggregated_warnings)])
    max_warning_name_length = maximum([length(key[2]) for key in keys(aggregated_warnings)])
    max_description_length = 32
    header_warning_code = "Warning code"
    header_warning_name = "Warning name"
    header_description = "Warning description"

    # Ensure header fields are included in padding calculations
    max_warning_code_length = max(max_warning_code_length, length(header_warning_code))
    max_warning_name_length = max(max_warning_name_length, length(header_warning_name))
    max_description_length = max(max_description_length, length(header_description))

    # Print header
    println("-"^(max_warning_code_length + max_warning_name_length + max_description_length + 29)) # Adjust total length based on the columns and separators
    println("$(rpad(header_warning_code, max_warning_code_length)) | $(rpad(header_warning_name, max_warning_name_length)) | $(rpad(header_description, max_description_length)) | Occurrences")
    println("-"^(max_warning_code_length + max_warning_name_length + max_description_length + 29))

    # Sort and print each warning with proper padding
    for key in sort(collect(keys(aggregated_warnings)))
        warning_data = aggregated_warnings[key]
        data_str = join(["($(data[1]), $(data[2]))" for data in warning_data if data[1] != ""], " ")
        data_str = data_str == "" ? "$(warning_data[1][2])" : data_str  # Handle no specific data case

        println("$(rpad(key[1], max_warning_code_length)) | $(rpad(key[2], max_warning_name_length)) | $(rpad(key[3], max_description_length)) | $data_str")
    end
    println("-"^(max_warning_code_length + max_warning_name_length + max_description_length + 29))
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

    print_warnings_dict(dict)
end