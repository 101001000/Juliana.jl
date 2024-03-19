function function_name(expr)
    if !(expr isa Expr) || expr.head != :function
        error("Trying to retrieve the function name of an unkown object")
    else
        return expr.args[1].args[1]
    end
end

function uncurlyfy(expr)
    if typeof(expr) != Expr
        return expr
    end
    if expr.head == Symbol("curly")
        return expr.args[1]
    end
    return expr
end

function expr_to_string(expr)
    io = IOBuffer()
    Base.show_unquoted(io, expr, 0, -1)
    return String(take!(io))
end

function expr_list_to_string(list)
    str = ""
    for el in list
        str *= expr_to_string(el) * "\n"
    end
    return str
end


function expr_identify(expr, str)
    if typeof(expr) != Expr
        return false
    end
    return expr_to_string(expr) == str
end

function expr_identify_1(expr, str)
    try    
        return expr_to_string(expr.args[1]) == str
    catch
        return false
    end
end

function expr_identify_1_1(expr, str)
    try    
        return expr_to_string(expr.args[1].args[1]) == str
    catch
        return false
    end
end

function expr_identify_any(expr, str)
    res = false
    try
        for arg in expr.args
            if expr_to_string(arg) == str
                res = true
            end
            res |= expr_identify_any(arg, str)
        end
    catch
    end
    return res
end

function expr_identify_line(expr, str)
    if typeof(expr) == Expr
        return expr.head != Symbol("block") && expr_identify_any(expr, str) && expr.head != Symbol("function") && expr.head != Symbol("if") 
    end
    return false
end

function expr_identify_n(expr, str)
    try
        for arg in expr.args
            if expr_identify_any(arg, str)
                return true
            end
        end
    catch
    end
    return false
end



function expr_has_symbol(expr, sym)
    if typeof(expr) == Symbol
        return sym == expr
    else
        has_symbol = false
        for arg in expr.args
            has_symbol |= expr_has_symbol(arg, sym)
        end
        return has_symbol
    end
end

function block_cleaner!(expr)
    if expr isa Expr
        for i in eachindex(expr.args)
            if expr.args[i] isa Expr
                if expr.args[i].head == :block && expr.head == :block
                    indices = []
                    for j in eachindex(expr.args[i].args)
                        if expr.args[i].args[j] isa LineNumberNode
                            continue
                        end
                        push!(indices, j)
                    end
                    if length(indices) == 0
                        expr.args[i] = LineNumberNode(1)
                    end

                    if length(indices) == 1
                        expr.args[i] = expr.args[i].args[indices[1]]
                    end
                end
            end
            block_cleaner!(expr.args[i])
        end
    end
end

