library(CVXR)

l2_loss = function(x, y, beta){
  return (norm2(x %*% beta - y)**2 /2)
}

poisson_loss = function(x, y, beta){
  n = dim(x)[1]
  return (sum(exp(x %*% beta) - multiply(y,x %*% beta)))
}

logit_loss = function(x, y, beta){
  n = dim(x)[1]
  loglik = sum(multiply(y,x %*% beta) - logistic(x %*% beta))
  return (-loglik / n)
}