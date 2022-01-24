matrix2latex <- function(mat, capture=TRUE) {
  matrix2latex <- function(mat) {
    printmrow <- function(x) {
      cat(cat(x,sep=" & "),"\\\\ \n")
    }
    cat("\\begin{bmatrix}","\n")
    body <- apply(mat,1,printmrow)
    cat("\\end{bmatrix}")
  }
  if (capture) {
    output <- paste0(capture.output(matrix2latex(mat)), collapse = "")
    return(output)
  } else {
    matrix2latex(mat)
  }
}
