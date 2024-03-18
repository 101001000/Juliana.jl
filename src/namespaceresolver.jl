include("exprutils.jl")

# TODO: rethink the architecture of this.

# Get an AST and if the namespace CUDA is on it, make it explicit over all calls
function explicit_using_replace!(expr)

    if typeof(expr) != Expr
        return
    end

    for i in eachindex(expr.args)

        arg = expr.args[i]

        if typeof(arg) != Expr
            continue
        end

        if arg.head == Symbol("call") || arg.head == Symbol("macrocall") || arg.head == Symbol("curly")
            if function_call_is_in_cuda_namespace(arg)
                expr.args[i] = add_namespace(arg, "CUDA") 
            end
        end
        explicit_using_replace!(expr.args[i])
    end
end

function function_call_is_in_cuda_namespace(expr)
    @assert typeof(expr) == Expr
    @assert expr.head == Symbol("call") || expr.head == Symbol("macrocall") || expr.head == :curly
    n = names(CUDA)
    return uncurlyfy(expr.args[1]) in n
end

function add_namespace(expr, namespace)
    if typeof(expr) == Symbol
        return Expr(Symbol("."), Symbol(namespace), QuoteNode(expr))
    else
        new_expr = copy(expr)

        if typeof(new_expr.args[1]) == Expr
            new_expr.args[1].args[1] = Expr(Symbol("."), Symbol(namespace), QuoteNode(new_expr.args[1].args[1]))
        else
            new_expr.args[1] = Expr(Symbol("."), Symbol(namespace), QuoteNode(new_expr.args[1]))
        end
        return new_expr
    end
end



# using MyModule
# in-scope: All exported names (x and y), MyModule.x, MyModule.y, and MyModule.p
# extensible: MyModule.x, MyModule.y, and MyModule.p
# using MyModule: x, p
# in-scope: x and p
# extensible: (nothing)
# import MyModule
# in-scope: MyModule.x, MyModule.y, and MyModule.p
# extensible: MyModule.x, MyModule.y, and MyModule.p
# import MyModule.x, MyModule.p
# in-scope: x and p
# extensible: x and p
# import MyModule: x, p
# in-scope: x and p
# extensible: x and p


# Return a sorted list of the global imports. TODO
#function extract_modules(top_expr)
#    return["CUDA"]
#end
#
#function get_namespace(top_expr, global_symbol)
#    using_modules = extract_modules(top_expr)
#    namespace = nothing
#    for modul in using_modules
#        names = names(modul)
#        for name in names
#            if global_symbol == Symbol(name)
#                namespace = name
#            end
#        end
#    end
#end
