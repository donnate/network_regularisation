source("src/cd_solver.R")
source("src/penalties.R")
source("src/losses.R")

genglm <- function(x, y, D, family = c("gaussian", "binomial", "poisson"),
                   lambda1 = 1.,
                   lambda2 = 1.,
                   standardize = TRUE,
                   solver = c("ECOS", "CD"),
                   intercept = TRUE){
  
  n = nrow(x)
  p = ncol(x)
  beta1 = Variable(p)
  
  if (family == "gaussian"){
    if (solver == "ECOS"){
      prob1 = Problem(Minimize(l2_loss(x, y, beta1) +
                                 gen_penalty(beta1, 
                                             lambda1,
                                             lambda2,
                                             D)))
      CVXR_solution <- solve(prob1)
      hat_beta  = CVXR_solution$getValue(beta1)
      return(hat_beta)
    }else{
      if (solver == "CD"){
        hat_beta  = cgd_solver(x ,y, D, lambda1, lambda2, eps = 1e-4, max_it = 10000)
        return(hat_beta)
      }else{
        print("Not implemented yet")
        return(NA)
      }
    }
  }
  if (family == "binomial"){
    if (solver == "ECOS"){
      print("Not implemented yet")
      return(NA)
    }
  }
  
  if (family == "poisson"){
    if (solver == "ECOS"){
      print("Not implemented yet")
      return(NA)
    }
  }
  
  
}
