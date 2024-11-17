using CUDA

function node_to_string(node)
	io = IOBuffer()
	show_unquoted(IOContext(io, :unquote_fallback => false), node, 0, -1)
	return String(take!(io))
end

function wrap_with_namespace(node)
	return Expr(:., :CUDA, QuoteNode(node))
end

function drop_ftype(expr)
	if @capture(expr, fname_{T__})
		return fname
	end
	return expr
end

function drop_module(expr)
	expr = drop_ftype(expr)
	if @capture(expr, M_.fname_)
		return fname
	end
	return expr
end


function drop_type(expr)
	if @capture(expr, v_::T_)
		return v
	else
		return expr
	end
end

function head_is(node, symbol)
	if node isa Expr
		if node.head == symbol
			return true
		end
	end
	return false
end


function capture_fdef(node)

	fname = nothing
	fargs = nothing
	fbody = nothing
	T = nothing

    @capture(node, 
        (function fname_(fargs__) fbody_ end) |
        (function fname_(fargs__) where {T__} fbody_ end) |
        (function fname_(fargs__) where T__ fbody_ end) |
		(function fname_{T1__}(fargs__) fbody_ end) |
        (function fname_{T1__}(fargs__) where {T__} fbody_ end) |
        (function fname_{T1__}(fargs__) where T__ fbody_ end) |
        (fname_(fargs__) = fbody_) |
        (fname_(fargs__) where {T__} = fbody_) |
        (fname_(fargs__) where T__ = fbody_)
	)
    
	return fname, fargs, fbody, T  
end

function capture_fdef_name(node)
	fname, _, _, _ = capture_fdef(node)
	return fname
end

function is_fdef(node)
	fname, _, _, _ = capture_fdef(node)
	return !isnothing(fname)
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
    dev = CUDA.device()
    atts = Dict()
    for CUatt in instances(CUDA.CUdevice_attribute_enum)
        try
            atts[CUatt] = attribute(dev, CUatt)
        catch
            @error string(CUatt) * " not available in this device"
        end
    end
    jsonString = JSON.json(atts)
	filepath = joinpath(@__DIR__, "..", "gpu-presets", replace(name(dev), " " => "_") * ".json")
	mkpath(dirname(filepath))
    open(filepath, "w") do file
        write(file, jsonString)
    end
	@info filepath * " file generated"
end



# This only works with function/macro calls.
# TODO: Make this work with already existing splatting...

function unsplat_fcallargs(func)
	if @capture(func, f_(arg_...))
		return Expr(func.head, f, arg...)
	end
	if @capture(func, f_(arg1_, arg2_...))
		return Expr(func.head, f, arg1, arg2...)
	end
	if @capture(func, f_(arg1_, arg2_, arg3_...))
		return Expr(func.head, f, arg1, arg2, arg3...)
	end
	if @capture(func, f_(arg1_, arg2_, arg3, arg4_...))
		return Expr(func.head, f, arg1, arg2, arg3, arg4...)
	end
	if @capture(func, @f_(arg_...))
		return Expr(:macrocall, f, LineNumberNode(0), arg...)
	end
	return func
end


# Quoting makes this anoying double body thing. 
function create_func(name, args, body)
	f = :(function $name($args...) $body end)
	f.args[2].args = f.args[2].args[3].args
	f.args[1] = unsplat_fcallargs(f.args[1])
	return f
end


#function Base.show_unquoted_expr_fallback(io::IO, ex::Expr, indent::Int, quote_level::Int)
#	if ex.head == Symbol(raw"$")
#		print(io, raw"$")
#		print(io, String(ex.args[1]))
#	else
#		print(io, "\$(Expr(")
#		Base.show(io, ex.head)
#		for arg in ex.args
#			print(io, ", ")
#			Base.show(io, arg)
#		end
#		print(io, "))")
#	end
#end
#
#function Base.valid_import_path(@nospecialize(ex), allow_as = true)
#    if allow_as && Base.is_expr(ex, :as) && length((ex::Expr).args) == 2
#        ex = (ex::Expr).args[1]
#    end
#	v1 = Base.is_expr(ex, :(.)) 
#	v2 = length((ex::Expr).args) > 0
#	v3 = all(a->isa(a,Symbol), (ex::Expr).args)
#	println("Checkiing")
#	println(v1)
#	println(v2)
#	println(v3)
#	println(ex.head)
#    return v1 && v2 && v3
#end

# Workaround for breaking seqs. I can make this with a regex and that will allow precompilation
#function Base.show_unquoted_expr_fallback(io::IO, ex::Expr, indent::Int, quote_level::Int)
#	if ex.head == :seq
#		for arg in ex.args
#			print(io, arg)
#			print(io, "\n")
#		end
#	else
#		# Normal base/show.jl behavior
#		print(io, "\$(Expr(")
#		show(io, ex.head)
#		for arg in ex.args
#			print(io, ", ")
#			show(io, arg)
#		end
#		print(io, "))")
#	end
#end