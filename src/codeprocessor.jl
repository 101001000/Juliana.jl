# Convert comments into an AST parsable structure
# TODO: add support for multiline comments
# TODO: fix comments separated with commas

function replace_comments(str)
    regex_pattern = r"#.*"
    replace_comment = match -> ";KAUtils.@comment \"\"\"$match\"\"\""
    modified_code = replace(str, regex_pattern => replace_comment; count=typemax(Int))
    return modified_code
end

function undo_replace_comments(str)
    regex_pattern = r"KAUtils\.@comment \"#(.*)\""
    modified_code = replace(str, regex_pattern => s"# \1"; count=typemax(Int))
    return modified_code
end

# Removes the expression structure arround interpolated expressions.
function replace_interpolation(str)
    pattern = r"\$\((Expr\(:\$, :\w+\))\)"
    inside_pattern = r"\$\(Expr\(:\$, :(.*?)\)\)"
    for m in eachmatch(pattern, str)
        new_str = match(inside_pattern, m.match).captures[1]
        str = replace(str, m.match => "\$" * new_str)
    end
    return str
end



