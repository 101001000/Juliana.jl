function node_to_string(node)
	io = IOBuffer()
	show_unquoted(IOContext(io, :unquote_fallback => false), node, 0, -1)
	return String(take!(io))
end

function wrap_with_namespace(node)
	return Expr(:., :CUDA, QuoteNode(node))
end


# Special prewalk function which will skip the walking by returning nothing
function skip_prewalk(f, node)
	new_node = f(node)
	if isnothing(new_node)
		return node
	end
    if new_node isa Expr
        new_args = [skip_prewalk(f, arg) for arg in new_node.args]
        return Expr(new_node.head, new_args...)
    elseif new_node isa Array
        return [skip_prewalk(f, elem) for elem in new_node]
    else
        return new_node
    end
end


# Special prewalk function which will skip the walking after applying the function
function prewalk_once(f, node)
	new_node = f(node)
	if new_node != node
		return new_node
	else
		if node isa Expr
			new_args = [prewalk_once(f, arg) for arg in node.args]
			return Expr(node.head, new_args...)
		elseif node isa Array
			return [prewalk_once(f, elem) for elem in node]
		else
			return node
		end
	end
end

# Special prewalk function which will skip the walking after applying the function
function prewalk_parent_info(f, node, parent)
	new_node = f(node, parent)
	if new_node isa Expr
        new_args = [prewalk_parent_info(f, arg, new_node) for arg in new_node.args]
        return Expr(new_node.head, new_args...)
    elseif node isa Array
        return [prewalk_parent_info(f, elem, new_node) for elem in new_node]
    else
        return new_node
    end
end

function dump_gpu_info()
    dev = device()
    atts = Dict()
    for CUatt in instances(CUDA.CUdevice_attribute_enum)
        try
            atts[CUatt] = attribute(dev, CUatt)
        catch
            @error string(CUatt) * " not available in this device"
        end
    end
    jsonString = JSON.json(atts)
    open("gpu-info/" * replace(name(dev), " " => "_") * ".json", "w") do file
        write(file, jsonString)
    end
end
