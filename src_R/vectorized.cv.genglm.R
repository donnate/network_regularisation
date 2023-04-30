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

#helper functions for vectorized.cv.genglm

# returns score for one fold 
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

# returns average mean squared error across folds for one lambda1 & lambda2 combination
param_cv = function(lambda_row, x, y, D, folds, family, solver,type.measure){
  l1 = lambda_row[1]
  l2 = lambda_row[2]
  param_score = mean(unlist(lapply(folds, folds_cv, x=x, y=y, D=D, family=family, solver=solver, l1=l1, l2=l2,type.measure=type.measure)))
  return(param_score)
}

vectorized.cv.genglm = function(formula = NULL, data = NULL, x = NULL, y = NULL, 
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