library(CVXR)

lasso_penalty = function(beta, l1){
  return (l1 * norm1(beta))
}
  
smoothlasso_penalty = function(beta, l1, l2, D){
  return (l1 * norm1(beta) + l2 * norm2(D %*% beta)**2)
}
  
elasticnet_penalty = function(beta, l1, l2){
  return (l1 * norm1(beta) + l2 * norm2(beta)**2)
}
  
fusedlasso_penalty = function(beta, l1, l2, D){
  return (l1 * norm1(beta) + l2 * norm1(D %*% beta))
}

ee_penalty = function(beta, l1, l2, D){
  return (l1 * norm1(D %*% beta) + l2*norm2(D %*% beta)**2)
}

gtv_penalty = function(beta, l1, l2, l3, D){
  return (l1 * norm1(D %*% beta) + l2*norm2(D %*% beta)**2 + l3 * norm1(beta))
}