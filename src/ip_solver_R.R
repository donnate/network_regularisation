library(roxygen2)
library(pracma)
library(roperators)
library(gear)

directions = function(D, u, y, lambda1, mu1, mu2, t1){
  m = dim(u)[1]
  f1 = matrix(u - lambda1 * rep(1, m), m, 1)
  f2 = matrix(- u - lambda1 * rep(1,m), m, 1)
  J1_inv = diag(c(mu1/f1), nrow = m)
  J2_inv = diag(c(mu2/f2), nrow = m)
  
  # Directions
  ##Solve A(d_u) = w
  t1 = c(t1)
  M = D %*% t(D)
  A_Cho = chol(M - J1_inv - J2_inv)
  w = -(M %*% u - D %*% y - 1/t1* 1/f1 + 1/t1 * 1/f2)
  d_u = solve_chol(A_Cho, w)
  
  
  d_mu1 = -(mu1 + 1/t1 * 1/f1 + J1_inv %*% d_u)
  d_mu2 = -(mu2 + 1/t1 * 1/f2 - J2_inv %*% d_u)
  return(list(d_u = d_u, d_mu1 = d_mu1, d_mu2 =  d_mu2))
}






line_search = function(D, u, y, lambda1, mu1, mu2, t1, d_u, d_mu1, d_mu2, a = 0.1, b = 0.7){
  m = dim(u)[1]
  f1 = matrix(u - lambda1 * rep(1, m), m, 1)
  f2 = matrix(- u - lambda1 * rep(1,m), m, 1)
  
  a0 = -mu1/d_mu1
  t1 = c(t1)
  a1 = c(a0[a0 >= 0], 1)             #avoid empty set
  b0 = -mu2/d_mu2
  b1 = c(b0[b0 >= 0], 1)
  
  s_max = min(1, min(a1), min(b1))
  s = 0.99 * s_max
  
  while (any(f1 + s* d_u >=0) & any(f2 - s* d_u >=0)){s = b*s}
  
  r_t_0 = residuals_1(D, u, y, lambda1, mu1, mu2, t1)
  
  while (residuals_1(D, u+s*d_u, y, lambda1, mu1+s*d_mu1, mu2+s*d_mu2, t1) > (1-a*s) * r_t_0){s = b *s}
  
  return (s)
}


residuals_1 = function(D, u, y, lambda1, mu1, mu2, t1){
  m = dim(u)[1]
  f1 = matrix(u - lambda1 * rep(1, m), m, 1)
  f2 = matrix(- u - lambda1 * rep(1,m), m, 1)
  # residuals
  t1 = c(t1)
  M = D %*% t(D)
  r1 = M %*% u - D %*% y + mu1 - mu2
  r2 = -diag(c(mu1), nrow = m) %*% f1 - 1/t1 * rep(1,m)
  r3 = -diag(c(mu2), nrow = m) %*% f2 - 1/t1 * rep(1,m)
  
  r_t = norm(cbind(r1, r2, r3), '2')
  
  return(r_t)
}



s_gap = function(u, lambda1, mu1, mu2){
  m = dim(u)[1]
  f1 = u - lambda1 * rep(1, m)
  f2 = - u - lambda1 * rep(1,m)
  f1 = matrix(f1, m, 1)
  f2 = matrix(f2, m, 1)
  return(-t(f1) %*% mu1 - t(f2) %*% mu2)
}



ip_solver = function(X,y, D, lambda1, lambda2, mu = 1.5, eps = 1e-4, max_it = 10000){
  m = dim(D)[1]
  p = dim(D)[2]
  X_til = rbind(X, sqrt(2*lambda2)*D)
  y_til = rbind(y, matrix(0, m, 1))
  X_til_pinv = pinv(X_til)
  y_v = X_til %*% (X_til_pinv %*% y_til)
  D_v = D %*% X_til_pinv
  
  
  u = matrix(0, m, 1)
  mu1 = matrix(10*rep(1,m), ncol = 1)
  mu2 = matrix(10*rep(1,m), ncol = 1)
  
  t1 = 2*m*mu/s_gap(u, lambda1, mu1, mu2)
  
  n_iter = 0
  
  
  while (TRUE){
    n_iter %+=% 1
    if (n_iter >= max_it){
      #raise ValueError("Iterations exceed max_it")
      print("Iterations exceed max_it")
      return (X_til_pinv %*% (y_v - t(D_v) %*% u))
    }
    ds_result = directions(D_v, u, y_v, lambda1, mu1, mu2, t1)
    d_u= ds_result$d_u
    d_mu1 = ds_result$d_mu1
    d_mu2 =ds_result$d_mu2
    s = line_search(D_v, u, y_v, lambda1, mu1, mu2, t1, d_u, d_mu1, d_mu2)
    
    u %+=% (s*d_u)
    mu1 %+=% (s*d_mu1)
    mu2 %+=% (s*d_mu2)
    
    
    r_t = residuals_1(D_v, u, y_v, lambda1, mu1, mu2, t1)
    eta = s_gap(u, lambda1, mu1, mu2)
    
    t1 = 2*m*mu/eta   # 2m since we have mu1 and mu2 of total 2m variables
    
    
    if (r_t <= eps & eta <= eps){break}
  }  
  
  beta = X_til_pinv %*% (y_v - t(D_v) %*% u)
  return (beta)
}
