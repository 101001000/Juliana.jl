include("exprutils.jl")

function kernelize_function!(expr, sym, fs, inliner_it)
    for i in eachindex(expr.args)
        if typeof(expr.args[i]) != Expr 
            continue
        end
        if expr.args[i].head == :function
            if(expr.args[i].args[1].args[1] == sym)
                remove_farg_types!(expr.args[i])

                var_ids = Set()

                extract_const_vars(expr.args[i], var_ids)
                constantify_function!(expr.args[i], var_ids)

                # TODO: Just one iteration is enough to translate all
                j = 1
                while inliner_it == -1 || j <= inliner_it
                    ast_pre_inline = copy(expr.args[i].args[2])
                    function_call_inliner!(expr.args[i].args[2], fs)

                    if (ast_pre_inline == expr.args[i].args[2]) # No more changes required.
                        break
                    end
                    j += 1
                end

                expr.args[i] = Expr(:macrocall, Symbol("@kernel"), LineNumberNode(1), expr.args[i])
                continue
            end
        end
        kernelize_function!(expr.args[i], sym, fs, inliner_it)
    end
end

function remove_farg_types!(expr)
    for i in eachindex(expr.args[1].args)
        if typeof(expr.args[1].args[i]) != Expr 
            continue
        end
        if expr.args[1].args[i].head == Symbol("::")
            expr.args[1].args[i] = expr.args[1].args[i].args[1]
        end
    end
end


function extract_const_vars(expr, s)
    if expr isa Expr
        for arg in expr.args
            if expr_identify_1(arg, "CUDA.ldg")
                push!(s,arg.args[2])
                continue
            end
            extract_const_vars(arg, s)
        end
    end
    return s
end



function constantify_function!(expr, args_ids)
    for i in eachindex(expr.args[1].args[2:end])
        if expr.args[1].args[i] in args_ids
            expr.args[1].args[i] = Expr(:macrocall, Symbol("@Const"), LineNumberNode(1), expr.args[1].args[i])
        end
    end
end



# It will replace all the returns with the correspondent @goto.
function return_replacer!(expr, retname, retlabel)
    if typeof(expr) == Expr
        for i in eachindex(expr.args)
            if typeof(expr.args[i]) != Expr 
                continue
            end
            if expr.args[i].head == :return
                ass_expr = Expr(Symbol("="), Symbol(retname), expr.args[i].args[1])
                goto_expr = Meta.parse("@goto " * retlabel)
                expr.args[i] = Expr(:block, ass_expr, goto_expr)
            end
            return_replacer!(expr.args[i], retname, retlabel)
        end
    end
end


function drop_type(expr)
    if !(expr isa Expr)
        return expr
    end
    if expr.head == Symbol("::")
        return expr.args[1]
    end
    return expr
end

# if applied to an expr, it will replace all the function calls with the definitions stored in the fs_inlined keys.
# fs_inlined stores the times certain function has been inlined
# because kernels cannot have return statements, those are replaced with @goto and @label macros, adding a return variable at the end of the let construct
function function_call_inliner!(expr, fs_inlined)
    for i in eachindex(expr.args)
        if typeof(expr.args[i]) != Expr 
            continue
        end
        if expr.args[i].head == :call
            
            #check if the function call is in the list of functions extracted.
            for f in keys(fs_inlined)
                if f.args[1].args[1] == expr.args[i].args[1]

                    new_body = copy(f.args[2])
                    
                    # iterate over all the function definition arguments
                    for j in eachindex(f.args[1].args)
                        if j == 1 # we ignore the function call
                            continue
                        end
                        # replace the function definition arguments for the call ones.
                        expr_replace!(new_body, drop_type(f.args[1].args[j]), expr.args[i].args[j])
                    end

                    ret_name = string(f.args[1].args[1]) * "_return_value" * string(fs_inlined[f])
                    label_name = string(f.args[1].args[1]) * "_end_" * string(fs_inlined[f])
                    ret_init = Expr(Symbol("="), Symbol(ret_name), :nothing)
                    ret_end = Meta.parse("@label " * label_name)

                    return_replacer!(new_body, ret_name, label_name)
                    fs_inlined[f] += 1
                    expr.args[i] = Expr(:let, Expr(:block), Expr(:block, ret_init, new_body, ret_end, Symbol(ret_name)))
                end
            end 
        end
        function_call_inliner!(expr.args[i], fs_inlined)
    end
end