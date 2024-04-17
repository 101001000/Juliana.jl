using JSON, CSV, DataFrames

function normalize_csvs(csv1, csv2)

    df1 = DataFrame(CSV.File(csv1))
    df2 = DataFrame(CSV.File(csv2))

    for row1 in eachrow(df1)
        processed_row = false
        for row2 in eachrow(df2)
            if row1["benchmark"] == row2["benchmark"]
                nf = row1["accmedian"]
                row1["accmedian"] /= nf
                row2["accmedian"] /= nf
                processed_row = true
            end
        end

        if !processed_row
            println("Row without match, " * string(row1["benchmark"]))
        end
    end

    CSV.write(csv1, df1)
    CSV.write(csv2, df2)

end

function main()
    normalize_csvs(ARGS[1], ARGS[2])
end

main()
