source("src_R/cd_solver_R.R")
source("src_R/ip_solver_R.R")
source("src_R/admm_solver_R.R")
source("src_R/penalties.R")
source("src_R/losses.R")
source("src_R/genglm.R")
source("src_R/param_cv.R")
source("src_R/folds_cv.R")
library(caret)
library(tidyverse)
library(pracma)
library(roperators)
library(MASS)
library(caret)

#' Cross-validation for genglm
#' @param formula formula for model
#' @param data data matrix containing input matrix and response variable
#' @param x input matrix
#' @param y response variable
#' @param D adjacency matrix 
#' @param family distribution family
#' @param solver optimization solver
#' @param type.measure mean squared error or deviance
#' @param lambda1 list of values for lambda 1
#' @param lambda2 list of values for lambda 2
#' @param n_folds number of folds
#' @return cv.genglm object

cv.genglm = function(formula = NULL, data = NULL, x = NULL, y = NULL, 
                                D,
                                type.measure = c("deviance","mean_squared_error"),
                                family = c("gaussian","binomial","poisson"),
                                solver = c("ECOS", "IP", "ADMM", "CD"),
                                lambda1 = 10^seq(-3,2,1), lambda2 = 10^seq(-3,2,1),
                                n_folds=2){
  
  if(!is.null(formula) & !is.null(data)){
    vars = all.vars(formula)
    response = vars[1]
    predictors = vars[-1]
    design_matrix = model.matrix(formula,data)
    x = as.matrix(design_matrix[,-1])
    y = as.matrix(data[response])
  }
  
  N <- nrow(x)
  folds = createFolds(1:N,k=n_folds)
  
  lambdas = expand.grid(lambda1,lambda2)
  colnames(lambdas) = c('l1','l2')
  cv_scores = apply(lambdas, 1, param_cv, x=x,y=y,D=D,folds,family=family,solver=solver,type.measure=type.measure)
  
  lambda1.min = lambdas[which.min(cv_scores),]$l1
  lambda2.min = lambdas[which.min(cv_scores),]$l2
  
  model.min = genglm(x=x, y=y,
                     D=D,family=family,
                     lambda1=lambda1.min,
                     lambda2=lambda2.min, solver=solver,
                     standardize = TRUE)
  hat_beta.min = model.min$hat_beta
  residuals = y - x%*%model.min$hat_beta
  
  cv <- list(lambda1.min = lambda1.min, lambda2.min = lambda2.min, 
             family = family, solver = solver,
             hat_beta.min = hat_beta.min,
             residuals = residuals)
  attr(cv, "class") <- "cv.genglm"
  cv
}