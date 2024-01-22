function to_year(str::String)
    η = contains(str, "MO") ? 1 / 12 : 1
    val = parse(Float64, replace(str, " MO" => "", " YR" => ""))
    return val * η
end