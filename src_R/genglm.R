setwd("~/Desktop/network_regularization/network_regularisation")

source("src_R/cd_solver_R.R")
source("src_R/ip_solver_R.R")
source("src_R/admm_solver_R.R")
source("src_R/penalties.R")
source("src_R/losses.R")
library(caret)

genglm <- function(x, y, D, family = c("gaussian", "binomial", "poisson"),
                   lambda1 = 1.,
                   lambda2 = 1.,
                   standardize = TRUE,
                   solver = c("ECOS", "IP", "ADMM", "CD"),
                   intercept = TRUE){
  
  n = nrow(x)
  p = ncol(x)
  beta1 = Variable(p)
  
  if(family != "gaussian" || is.null(solver)){
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
    #return(hat_beta)
  }
  else if(solver == "IP" && family == "gaussian"){
    hat_beta = ip_solver(x, y, D, lambda1, lambda2, mu = 1.5, eps = 1e-4, max_it = 10000)
    #return(hat_beta)
  }
  else if(solver == "ADMM" && family == "gaussian"){
    hat_beta = admm_solver(x, y, D, lambda1, lambda2, rho = 1, eps = 1e-3, max_it = 50000)
    #return(hat_beta)
  }
  else if(solver == "CD" && family == "gaussian"){
    hat_beta  = cgd_solver(x ,y, D, lambda1, lambda2, eps = 1e-4, max_it = 10000)
    #return(hat_beta)
  }
  else if (solver == "ECOS" && family == "gaussian"){
      prob1 = Problem(Minimize(l2_loss(x, y, beta1) + gen_penalty(beta1, lambda1,lambda2,D)))
      CVXR_solution <- solve(prob1)
      hat_beta  = CVXR_solution$getValue(beta1)
      #return(hat_beta)
  }
  else{
        print("Not implemented yet")
        #return(NA)
  }
  fit <- list(hat_beta = hat_beta, lambda1 = lambda1, lambda2 = lambda2, family = family, solver = solver)
  attr(fit, "class") <- "genglm"
  fit
}



predict.genglm <- function(object,newx,type=c("link","response"),exact=FALSE){
  
  # gaussian: return fitted values
  # binomial: return linear predictors if type==link, return fitted probabilities if type==response
  # poisson: return linear predictors if type==link, return fitted mean if type==response
  
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

coef.genglm <- function(object){
 return(object$hat_beta)
  #in glmnet, calls predict(object,type='coefficients')
  # a0=t(as.matrix(object$a0)) a0 is the sequence of intercepts for each of the lambdas tested
  # rownames(a0)="(Intercept)"
  # nbeta=methods::rbind2(a0,object$beta) combining intercepts and the betas
  # if we want to use a lambda that is different from the one in the model fit, do linear interpolation
  # if(!is.null(s)){
  #   vnames=dimnames(nbeta)[[1]]
  #   dimnames(nbeta)=list(NULL,NULL)
  #   lambda=object$lambda
  #   lamlist=lambda.interp(lambda,s)
  #   
  #   nbeta=nbeta[,lamlist$left,drop=FALSE]%*%Diagonal(x=lamlist$frac) +nbeta[,lamlist$right,drop=FALSE]%*%Diagonal(x=1-lamlist$frac)
  #   namess=names(s)
  #   if(is.null(namess))namess=paste0("s",seq(along=s))
  #   dimnames(nbeta)=list(vnames,namess)
  # }
}


cv.genglm <- function(x,
                      y,
                      D,
                      type.measure = c("deviance","mean_squared_error"),
                      family = c("gaussian","binomial","poisson"),
                      solver = c("ECOS", "IP", "ADMM", "CD"),
                      lambda1 = 10^seq(-3,2,1), lambda2 = 10^seq(-3,2,1),
                      n_folds=2){
  # lambda1 and lambda2 are lists of the candidate values for lambda1 and lambda2
  #model.cv = cv.genglm(X,y,D,family='gaussian',solver='CD')
  print(c(family, solver))
  N <- nrow(x)
  folds = createFolds(1:N,k=n_folds)
  
  lambdas = expand.grid(lambda1,lambda2)
  colnames(lambdas) = c('l1','l2')
  cv_scores = c()
  
  for(i in 1:nrow(lambdas)){
    l1 = lambdas[i,]$l1
    l2 = lambdas[i,]$l2
    
    fold_scores = c()
    for(fold in folds){
      train_x = x[-fold,]
      test_x = x[fold,]
      
      train_y = matrix(y[-fold,])
      test_y = matrix(y[fold,])
      
      model = genglm(x=train_x, y=train_y,
                     D=D,family=family,
                     lambda1=l1,
                     lambda2=l2, solver=solver,
                     standardize = TRUE)
      fitted_y = predict.genglm(model, test_x,type="response")
      
      # if(type.measure == "deviance"){
      #   
      # }
      fold_scores = c(fold_scores, mean((test_y-fitted_y)^2))
    }
    cv_scores = c(cv_scores, mean(fold_scores))
  }
  
  lambda1.min = lambdas[which.min(cv_scores),]$l1
  lambda2.min = lambdas[which.min(cv_scores),]$l2
  
  # lambda1.1se
  # lambda2.1se
  
  cv <- list(lambda1.min = lambda1.min, lambda2.min = lambda2.min, family = family, solver = solver)
  attr(cv, "class") <- "cv.genglm"
  cv
}

