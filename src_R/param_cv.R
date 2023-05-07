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
#' @param lambda_row lambda 1 and lambda 2
#' @param x input matrix
#' @param y response variable
#' @param D adjacency matrix 
#' @param folds folds
#' @param family distribution family
#' @param solver optimization solver
#' @param type.measure mean squared error or deviance
#' @return average score across folds for one lambda 1 & lambda 2 combination

param_cv = function(lambda_row, x, y, D, folds, family, solver,type.measure){
  l1 = lambda_row[1]
  l2 = lambda_row[2]
  param_score = mean(unlist(lapply(folds, folds_cv, x=x, y=y, D=D, family=family, solver=solver, l1=l1, l2=l2,type.measure=type.measure)))
  return(param_score)
}
