source("src_R/cd_solver_R.R")
source("src_R/ip_solver_R.R")
source("src_R/admm_solver_R.R")
source("src_R/penalties.R")
source("src_R/losses.R")
library(caret)
library(tidyverse)
library(pracma)
library(roperators)
library(MASS)

#' Create fitted genglm object
#' @param formula formula for model
#' @param data data matrix containing input matrix and response variable
#' @param x input matrix
#' @param y response variable
#' @param D adjacency matrix 
#' @param family distribution family
#' @param lambda1 list of values for lambda 1
#' @param lambda2 list of values for lambda 2
#' @param solver 
#' @return fitted genglm object

genglm <- function(formula = NULL, data = NULL, x = NULL, y = NULL, 
                   D, family = c("gaussian", "binomial", "poisson"),
                   lambda1 = 1.,
                   lambda2 = 1.,
                   solver = c("ECOS", "IP", "ADMM", "CD")){
  
  if(!is.null(formula) & !is.null(data)){
    vars = all.vars(formula)
    response = vars[1]
    predictors = vars[-1]
    design_matrix = model.matrix(formula,data)
    x = as.matrix(design_matrix[,-1])
    y = as.matrix(data[response])
  }
  
  n = nrow(x)
  p = ncol(x)
  
  if(family != "gaussian" || is.null(solver)){
    beta1 = Variable(p)
    if(family == "binomial"){
      if (solver == "ECOS"){
        prob1 = Problem(Minimize(logit_loss(x, y, beta1)+gen_penalty(beta1,lambda1,lambda2,D)))
      }
    }
    if (family == "poisson"){
      if (solver == "ECOS"){
        prob1 = Problem(Minimize(poisson_loss(x, y, beta1)+gen_penalty(beta1,lambda1,lambda2,D)))
      }
    }
    if(family == "gaussian"){
      prob1 = Problem(Minimize(l2_loss(x, y, beta1)+gen_penalty(beta1,lambda1,lambda2,D)))
    }
    CVXR_solution <- solve(prob1)
    hat_beta = CVXR_solution$getValue(beta1)
  }
  else if(solver == "IP" && family == "gaussian"){
    hat_beta = ip_solver(x, y, D, lambda1, lambda2, mu = 1.5, eps = 1e-4, max_it = 10000)
  }
  else if(solver == "ADMM" && family == "gaussian"){
    hat_beta = admm_solver(x, y, D, lambda1, lambda2, rho = 1, eps = 1e-3, max_it = 50000)
  }
  else if(solver == "CD" && family == "gaussian"){
    hat_beta  = cgd_solver(x ,y, D, lambda1, lambda2, eps = 1e-4, max_it = 10000)
  }
  else if (solver == "ECOS" && family == "gaussian"){
    beta1 = Variable(p)
    prob1 = Problem(Minimize(l2_loss(x, y, beta1) + gen_penalty(beta1, lambda1,lambda2,D)))
    CVXR_solution <- solve(prob1)
    hat_beta  = CVXR_solution$getValue(beta1)
  }
  else{
    print("Not implemented yet")
    #return(NA)
  }
  
  residuals = y - x %*% hat_beta
  fit <- list(hat_beta = hat_beta, lambda1 = lambda1, lambda2 = lambda2, family = family, solver = solver, residuals = residuals)
  attr(fit, "class") <- "genglm"
  fit
}