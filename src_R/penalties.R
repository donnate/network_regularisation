library(CVXR)


lasso_penalty <- function(beta, l1){
  return(l1 * p_norm(beta, 1))
}


smoothlasso_penalty <- function(beta, l1, l2, D){
  return(l1 * p_norm(beta, 1) + l2 * p_norm(D %*% beta, 2)^2)
}

elasticnet_penalty <- function(beta, l1, l2){
  return(l1 *  p_norm(beta, 1) + l2 * p_norm(beta, 2)^2)
}

fusedlasso_penalty<- function(beta, l1, l2, D){
  return(l1 *  p_norm(beta, 1) + l2 *  p_norm(D %*%  beta, 1))
}

gen_penalty <- function(beta, l1, l2, D){
  return(l1 * p_norm(D %*% beta, 1) + l2 * p_norm(D %*%  beta, 2)^2)
}


gtv_penalty <- function(beta, l1, l2, l3, D){
  return(l1 * p_norm(D %*% beta, 1) + l2* p_norm(D %*% beta, 2)^2 + l3 * p_norm(beta, 1))
}

