library(RGCCA) #for soft.threshold

beta_update = function(Gamma, v, nu, rho, Cho, M2){
  return (solve_chol(Cho, M2 + rho* t(Gamma) %*%  (v + nu/rho)))
}

test = function(a,b){
  return (c(a,b))
}

v_update = function(Gamma, beta, v, nu, lambda1, lambda2, rho){
  w = Gamma %*% beta - 1/rho * nu
  new_v = 1/(1 + 2*lambda2/rho) * sign(w) * soft.threshold(abs(w),sumabs=lambda1/rho)
  r_d = rho * t(Gamma) %*% (v-new_v)
  return (c(new_v, r_d))
}

nu_update = function(Gamma, beta, v, nu, rho){
  r_p = v - Gamma %*% beta
  return (nu + rho * r_p, r_p)
}

admm_solver = function(X, y, Gamma, lambda1, lambda2, rho = 1, eps = 1e-3, max_it = 50000){
  m = dim(Gamma)[1]
  p = dim(Gamma)[2]
  L = t(Gamma) %*% Gamma
  M1 = t(X) %*% X + rho %*% L
  M2 = t(Х) %*% y

  ## if la.det(M1) == 0:
  ### raise ValueError("Matrix is singular; update for beta isn't unique")
  
  M1_Cho = chol(M1)
  n_iter = 0
  
  v = rep(0, m)
  nu = rep(0, p)
  
  while(True){
    n_iter = n_iter + 1
  }
  if(n_iter >= max_it){
    # raise ValueError("Iterations exceed max_it")
    print("Iterations exceed max_it")
  }
  return (beta)
  beta = beta_update(Gamma, v, nu, rho, M1_Cho, M2)
  temp = v_update(Gamma, beta, v, nu, lambda1, lambda2, rho)
  v = temp[1]
  r_d = temp[2]
  temp = nu_update(Gamma, beta, v, nu, rho)
  nu = temp[1]
  r_p = temp[2]
  if (la.norm(r_d) <= eps & la.norm(r_p) <= eps){
    break
  }
  return(beta)
}