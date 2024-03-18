struct Warning
    message::String
    details::Dict{String, Any}
end

const WARNINGS = Dict{String, WarningDetails}()

function add_warning!(id::String, message::String, details::Dict{String, Any}=Dict())
    WARNINGS[id] = WarningDetails(message, details)
end

function emit_warning(warnings_collected::Vector{WarningDetails}, id::String)
    if haskey(WARNINGS, id)
        push!(warnings_collected, WARNINGS[id])
    else
        error("Warning ID not found: $id")
    end
end

function display_warnings(warnings_collected::Vector{WarningDetails})
    for warning in warnings_collected
        println("Warning: $(warning.message)")
        for (key, value) in warning.details
            println("  $key: $value")
        end
    end
end


add_warning!("W001", "Use of deprecated function", Dict("line" => 42, "function_name" => "oldFunc"))
add_warning!("W002", "Variable is defined but not used", Dict("line" => 56, "variable_name" => "unusedVar"))

# Collect warnings
collected_warnings = Vector{WarningDetails}()

# Emit some warnings
emit_warning(collected_warnings, "W001")
emit_warning(collected_warnings, "W002")

# Display all collected warnings
display_warnings(collected_warnings)
