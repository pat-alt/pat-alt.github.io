# Helper functions:
∑(vector)=sum(vector)

# Compute cartesian product over two vectors:
function expandgrid(x,y)
    N = length(x) * length(y)
    grid = Iterators.product(x,y) |>
        Iterators.flatten |>
        collect |>
        z -> reshape(z, (2,N)) |>
        transpose |>
        Matrix
    return grid
end


