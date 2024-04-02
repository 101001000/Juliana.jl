using JSON, CSV, DataFrames

function compute_overhead!(df)
    for row1 in eachrow(df)
        processed_row = false
        for row2 in eachrow(df)
            if row1 == row2
                continue
            end
            if row1["kernelname"] == row2["kernelname"]
                nf = row1["mean"] / 100
                row1["overhead"] = (row1["mean"] - row2["mean"]) / nf
                row2["overhead"] = (row2["mean"] - row1["mean"]) / nf
                processed_row = true
            end
        end

        if !processed_row
            println("Row without match, " * string(row1["kernelname"]))
        end
    end

end

function generate_csv(files, output)
    df = DataFrame()
    added_columns = false
    for triple in files
        benchmark = triple.benchmark
        filepath = triple.filepath
        backend = triple.backend
        kernel_name = splitext(basename(filepath))[1]
        kernel_name = replace(kernel_name, "_" => "-")

        if isfile(filepath)
            jsondata = JSON.parsefile(filepath)
            if !added_columns
                df[!, :benchmark] = Vector{String}()
                df[!, :kernelname] = Vector{String}()
                df[!, :backend] = Vector{String}()
                for (key, value) in jsondata
                    df[!, Symbol(key)] = Vector{typeof(value)}()
                end
                df[!, :overhead] = Vector{Float32}()
                added_columns = true
            end
            row = [benchmark, kernel_name, backend, values(jsondata)... , 0.0]
            push!(df, row)
        else
            println("Error, " * string(triple) * " not found")
        end
    end
    compute_overhead!(df)
    CSV.write(output, df)
end

function parse_triple(triple_str)
    parts = split(triple_str, ",")
    return (filepath=parts[1], benchmark=parts[2], backend=parts[3])
end

function main()
    triples_str = ARGS[1]
    triples_str = replace(triples_str, "\n" => "")
    triples_list = split(triples_str, ";")
    triples = [parse_triple(triple) for triple in triples_list]
    generate_csv(triples, "output.csv")
end

main()