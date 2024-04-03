using JSON, CSV, DataFrames

function generate_csv(files, output)
    df = DataFrame()
    added_columns = false
    for filepath in files

        benchmark_name = splitext(basename(filepath))[1]
        benchmark_name = replace(benchmark_name, "-aggregated" => "")

        if isfile(filepath)
            jsondata = JSON.parsefile(filepath)
            if !added_columns
                df[!, :benchmark] = Vector{String}()
                for (key, value) in jsondata
                    df[!, Symbol(key)] = Vector{typeof(value)}()
                end
                added_columns = true
            end
            row = [benchmark_name, values(jsondata)...]
            push!(df, row)
        else
            println("Error, " * string(filepath) * " not found")
        end
    end
    CSV.write(output, df)
end

function main()
    str = ARGS[2]
    str = replace(str, "\n" => "")
    str_list = split(str, ";")
    generate_csv(str_list, ARGS[1])
end

main()