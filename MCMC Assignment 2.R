##############################
### Obtaining initial Data ###
##############################

library(datasets)
library(MASS)
library(mvtnorm)
data(iris)
index_versicolor = which(iris[,5]=='versicolor')
index_virginica = which(iris[,5]=='virginica')
index_setosa = which(iris[,5]=='setosa')
iris[,5] = 1
iris[index_virginica,5] = 2
iris[index_setosa,5] = 3
iris[,5] = factor(iris[,5])
x <- iris[1:150, c("Sepal.Length", "Sepal.Width", "Species")]
load(file = "index_iris_2182973.Rdata")
number_train = 149
x_train = as.matrix(x[index[1:number_train],1:2])
y_train = x[index[1:number_train],3]
x_test = as.matrix(x[index[number_train+1],1:2])
y_test = x[index[number_train+1],3]

X_train = cbind(matrix(1, dim(x_train)[1], 1), x_train)
X_test = cbind(matrix(1, dim(x_test)[1], 1), x_test)

#####################################
### Creating Likelihood Functions ###
#####################################


identity_func <- function(X_vec, beta_vec){
  
  X_m <- matrix(X_vec, nrow = 1, ncol = 3)
  
  beta_M <- matrix(beta_vec, nrow = 3, ncol = 1)
  
  Z <- X_m %*% beta_M  #defining sigmoid function
  return(Z)
}

log_likelihood <- function(X, y, beta){
  
  l <- 0
  n <- dim(X)[1]
  
  for (i in 1:n){
    
    Z_1 <- identity_func(X[i,], beta[1:3])
    Z_2  <- identity_func(X[i,], beta[4:6])
    Z_3 <- identity_func(X[i,], beta[7:9])
    
    l <- l + ((y[i] == 1)*Z_1) + ((y[i] == 2)*Z_2) + ((y[i] == 3)*Z_3) - log(exp(Z_1) + exp(Z_2) + exp(Z_3))
    
    
  }
  return(l) }




#####################################
### Metropolis-Hastings Algorithm ###
#####################################



Metropolis_Hastings <- function(X, y, iterations, beta0, sigma_q) {
  acc <- 0
  betas <- array(NA, dim = c(iterations+1, 9))
  betas[1,] <- beta0
  
  for (i in 1:iterations) {
    beta_star <- mvrnorm(n = 1, betas[i,], sigma_q)
    
    
    p <- (log_likelihood(X, y, beta_star) + 
            log(dmvnorm(beta_star, mean = matrix(0,9,1), sigma = diag(9) * 100)) + 
            log(dmvnorm(betas[i, ], mean = beta_star, sigma = sigma_q)) ) - 
      log_likelihood(X, y, betas[i, ]) -
      log(dmvnorm(betas[i, ], mean = matrix(0,9,1), sigma = diag(9) * 100)) - 
      log(dmvnorm(beta_star, mean = betas[i,], sigma = sigma_q)) 
    
    
    if (runif(1) < min(1, exp(p))) {
      betas[i+1,] <- beta_star
      acc <- acc + 1
    } 
    else {
      betas[i+1,] <- betas[i,]
    }
    
  }
  
  
  print(acc/iterations)
  
  return(betas)  
}

####################
### First MH run ###
####################


#start with initial betas equal to 1 and a diagonal covariance matrix with 30 in the diagonal and 0s everywhere else

beta0 = c(1,1,1,1,1,1,1,1,1)


log_likelihood(X_train, y_train, beta0)

sigma_001 <- diag(9)*0.01 #0.232525
betas_gen_001 <- Metropolis_Hastings(X_train, y_train, iterations = 200000, beta0, sigma_001) 


plot(betas_gen_001[,3], xlim = c(1,200001))

par(mfrow = c(3,3))
for (i in 1:9){
  plot(betas_gen_001[,i], xlim = c(1,200001))
  
}
 #does not mix very well

first_beta_matrix <<-  betas_gen_001

#discard first 100,000 iterations as this is only used to estimate the covariance matrix

betas_first_run <- first_beta_matrix[100001:200001,]





#multiESS(betas1) #achieves ESS of 3500, above the 1000 requirement which indicates convergence

sigma_pi <- var(betas_first_run) #using these estimated values to calculate our sample covariance matrix
sigma_q = (2.38**2) * sigma_pi / 9 #this covariance matrix will be used in the next run

####################
### Tuned MH Run ###
####################


betas <- Metropolis_Hastings(X_train, y_train, iterations = 200000, beta0, sigma_q) 

betas_optimised <<- betas

par(mfrow = c(3,3))
for (i in 1:9){
  plot(betas_optimised[,i], xlim = c(1,200001))
}
par(mfrow = c(1,1))

library(mcmcse)
multiESS(betas_optimised)

burn_in <- 25000
betas_optimised_ex_burn_in <- betas_optimised[burn_in:75001,] #only need to run for 50000 after burn in to achieve 
multiESS(betas_optimised_ex_burn_in) # ESS of 1000

####################
### Final MH Run ###
####################


beta_final <- Metropolis_Hastings(X_train, y_train, iterations = 75000, beta0, sigma_q)

par(mfrow = c(3,3))
for (i in 1:9){
  plot(betas[,i], xlim = c(1,125001))
}

 beta_final_ex_burn <- beta_final[burn_in:75001,]
multiESS(beta_final_ex_burn)



par(mfrow = c(1,1))
hist(beta_final_ex_burn[,9])

##########################
### Using Test Dataset ###
##########################

# class_1_log_likelihood <- function(X_test, beta){
#   
#     
#     Z_1 <- identity_func(X[i,], beta[1:3])
#     Z_2  <- identity_func(X[i,], beta[4:6])
#     Z_3 <- identity_func(X[i,], beta[7:9])
#     
#     l <- (Z_1) - log(exp(Z_1) + exp(Z_2) + exp(Z_3))
#     
#   return(l) }


log_probs<- array(NA, dim = c(dim(beta_final_ex_burn)[1], 1))

for (i in 1:dim(beta_final_ex_burn)[1]){

  log_l <- log_likelihood(X_test, y_test, beta_final_ex_burn[i,])
  
  log_probs[i] <- log_l
  
}

hist(log_probs, xlab = 'Log Probability', ylab = 'Frequency', main = 'Log Probability of Test Set Being Class 1')


# estimate of expected probability

print(mean(exp(log_probs)))
