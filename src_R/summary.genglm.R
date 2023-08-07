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


#' Summarize the genglm object
#' @param object fitted genglm object
#' @return residuals and coefficients of the fitted genglm object

summary.genglm = function(object){
  return(list(residuals=summary(object$residuals),coefficients=object$hat_beta))
}