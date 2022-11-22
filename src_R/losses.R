library(CVXR)


l2_loss <- function(x, y, beta){
  return(p_norm(x %*% beta - y, 2)^2 / 2)
}

poisson_loss <- function(x, y, beta){
    n = nrow(x)
    return(sum(exp(x %*% beta) - multiply(y , x %*% beta)))   ##/n make l1 l2 large
}

logit_loss <- function(x, y, beta){
  n = nrow(x)
  loglik = sum(multiply(y, x %*% beta) - logistic(x %*%  beta))
  return(-loglik / n)
}