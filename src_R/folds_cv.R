source("src_R/cd_solver_R.R")
source("src_R/ip_solver_R.R")
source("src_R/admm_solver_R.R")
source("src_R/penalties.R")
source("src_R/losses.R")
source("src_R/genglm.R")
library(caret)
library(tidyverse)
library(pracma)
library(roperators)
library(MASS)
library(caret)

#' Helper function for cv.genglm function
#' @param fold fold
#' @param x input matrix
#' @param y response variable
#' @param D adjacency matrix 
#' @param family distribution family
#' @param l1 lambda 1
#' @param l2 lambda 2
#' @param solver optimization solver
#' @param type.measure mean squared error or deviance
#' @return score for the fold 

folds_cv = function(fold, x, y, D, family, l1, l2, solver, type.measure){
  train_x = x[-unlist(fold),]
  test_x = x[unlist(fold),]
  
  train_y = matrix(y[-unlist(fold),])
  test_y = matrix(y[unlist(fold),])
  
  model = genglm(x=train_x,y=train_y,D=D,family=family,lambda1=l1,lambda2=l2,solver=solver)
  fitted_y = predict.genglm(model, test_x,type="response")
  n = nrow(test_x)
  
  if(type.measure == 'mean_squared_error'){
    fold_score = mean((test_y-fitted_y)^2)
  }
  else if(type.measure == 'deviance' & family == 'binomial'){
    #deviance
    fold_score = (-2/n)*sum(test_y*log(fitted_y/(1-fitted_y)) + log(1-fitted_y))
  }
  else if(type.measure == 'deviance' & family == 'poisson'){
    fold_score = (-2/n)*sum(test_y*log(test_y/fitted_y) - (test_y-fitted_y))
  }
  return(fold_score)
}