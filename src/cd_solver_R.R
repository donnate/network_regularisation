library(roxygen2)
library(pracma)
library(roperators)
library(gear)

cgd_solver = function(X,y, D, lambda1, lambda2, eps = 1e-4, max_it = 10000){
  m = dim(D)[1]
  p = dim(D)[2]
  X_til = rbind(X, sqrt(2*lambda2)*D)
  y_til = rbind(y, matrix(0, m, 1))
  X_til_pinv = pinv(X_til)
  y_v = X_til %*% (X_til_pinv %*% y_til)
  D_v = D %*% X_til_pinv
  
  Q = D_v %*% t(D_v)
  b = D_v %*% y_v
  
  u = matrix(0, m, 1)
  n_iter = 0
  prev_u = 0
  
  while(TRUE){
    n_iter %+=% 1
    if(n_iter > max_it){
      print("Iterations exceed max_it")
      return(X_til_pinv %*% (y_v - t(D_v) %*% u))
    }
    
    for (i in 1:m){
      t = 1/Q[i,i] * (b[i] - dot(Q[i,][-c(i)], u[-c(i)]))
      u[i] = sign(t)*min(abs(t), lambda1)
    }
    
    if (norm(u - prev_u, '2') <= eps){
      break
    }
    prev_u <- u
  }
  beta = X_til_pinv %*% (y_v - t(D_v) %*% u)
  return (beta)
}