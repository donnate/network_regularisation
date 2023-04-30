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

predict.genglm <- function(object,newx,type=c("link","response"),exact=FALSE){
  
  # gaussian: return fitted values
  # binomial: return linear predictors if type==link, return fitted probabilities if type==response
  # poisson: return linear predictors if type==link, return fitted mean if type==response
  
  if(class(object)=='genglm'){
    if(type == "link"){
      return(newx %*% object$hat_beta)
    }
    else if(type == "response"){
      if(object$family == "gaussian"){
        return(newx %*% object$hat_beta)
      }
      else if(object$family == "binomial"){
        return(exp(newx %*% object$hat_beta)/(1+exp(newx %*% object$hat_beta)))
      }
      else if(object$family == "poisson"){
        return(exp(newx %*% object$hat_beta))
      }
    }
    return(NA)
  }
  else if(class(object)=='cv.genglm'){
    if(type == "link"){
      return(newx %*% object$hat_beta.min)
    }
    else if(type == "response"){
      if(object$family == "gaussian"){
        return(newx %*% object$hat_beta.min)
      }
      else if(object$family == "binomial"){
        return(exp(newx %*% object$hat_beta.min)/(1+exp(newx %*% object$hat_beta.min)))
      }
      else if(object$family == "poisson"){
        return(exp(newx %*% object$hat_beta.min))
      }
    }
    return(NA)
  }
}