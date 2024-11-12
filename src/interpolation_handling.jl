raw"""

In the first parsing step, all $ symbols will be processed as $(Expr(...)).
The issue is that when printing the AST, most structures will rely in the unhandled show case if they have such expression instead the expected one. To avoid most of this cases, we substitute this first order $(Expr(Symbol())) expressions with an inlined $symbol (merge_interp_symbol). This won't work for composite expressions, but I didn't find that case yet. After this, parsing the AST will return var"\$value" for the direct interpolated symbols.

We must then, replace such symbols in the parsed string with the corrected $value ones (replace_var_interpolation)


Finally, there are some 




"""