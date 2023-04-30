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

coef.genglm <- function(object){
  if(class(object)=='genglm'){
    return(object$hat_beta)
  }
  else if(class(object)=='cv.genglm'){
    return(object$hat_beta.min)
  }
}