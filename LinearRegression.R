## Load Data
# ------------------------------------------------
load("/Residen.RData")
resdata = data.frame(Residen)
str(resdata)

# group variables 
group0 = subset(resdata, select=c(108,109,1:4))     #dates variables
group1 = subset(resdata, select=c(108,109,5:12))    #financial variables
group2 = subset(resdata, select=c(108,109,13:31))   #economic variables in time lag1
group3 = subset(resdata, select=c(108,109,32:50))   #economic variables in time lag2
group4 = subset(resdata, select=c(108,109,51:69))   #economic variables in time lag3
group5 = subset(resdata, select=c(108,109,70:88))   #economic variabels in time lag4
group6 = subset(resdata, select=c(108,109,89:107))  #ecomonic variables in time lag5

library(corrgram)
corrgram(group0, order="HC",abs=TRUE, cor.method="pearson", text.panel=panel.txt,
         main="Correlation of dates variables")
corrgram(group1, order="HC",abs=TRUE, cor.method="pearson", text.panel=panel.txt,
         main="Correlation of physical and financial variables")
corrgram(group2, order="HC",abs=TRUE, cor.method="pearson", text.panel=panel.txt,
         main="Correlation of time lag1 economic variables")
corrgram(group3, order="HC",abs=TRUE, cor.method="pearson", text.panel=panel.txt,
         main="Correlation of time lag2 economic variables")
corrgram(group4, order="HC",abs=TRUE, cor.method="pearson", text.panel=panel.txt,
         main="Correlation of time lag3 economic variables")
corrgram(group5, order="HC",abs=TRUE, cor.method="pearson", text.panel=panel.txt,
         main="Correlation of time lag4 economic variables")
corrgram(group6, order="HC",abs=TRUE, cor.method="pearson", text.panel=panel.txt,
         main="Correlation of time lag5 economic variables")
# the correlation plots suggest that there are a lot multicollinearity in the dataset
#library(corrplot)
#cor_matrix = corrplot(cor(resdata), method="number", number.cex=0.6)


## Linear Regression Model 
# ------------------------------------------------
reslm <- lm(log(V104) ~ .-V105, data=resdata)
summary(reslm)

varImp(reslm)

# why using log transformation on actual sales prices
library(ggplot2)
ggplot(data = resdata, aes(x = V104)) + geom_density(fill="lightblue")       #no_log
shapiro.test(resdata$V104)

ggplot(data = resdata, aes(x = log(V104))) + geom_density(fill="lightblue")  #log transformation
shapiro.test(log(resdata$V104))

ggplot(data=resdata, aes(x=sqrt(V104))) + geom_density(fill="lightblue")     #sqrt transformation
shapiro.test(log(resdata$V104))


## Split data
# ------------------------------------------------
set.seed(1)
row.number <- sample(1:nrow(resdata), 0.8*nrow(resdata))
train <- resdata[row.number,]
test <- resdata[-row.number,]
dim(train)
dim(test)


## Backwards selection Linear Regression model
# ------------------------------------------------
start_time <- Sys.time()
# generating a maximum model hold out
max_model_holdout = lm(log(V104)~.-V105, data=train)
# fit a minimal adequate model
help(stepAIC)
library(MASS)
backwards_holdout = stepAIC(max_model_holdout, direction="backward", trace=0)
summary(backwards_holdout)

end_time <- Sys.time()
cat("Backwards selection holdout computational time is", end_time-start_time, "seconds")

#library(tictoc)
#tic()
#toc()

# test and train MSE
library(Metrics)
pretest_back <- predict(backwards_holdout, test)
testmse_back <- mse(test$V104, exp(pretest_back))
cat("Backwards selection holdout test MSE is", testmse_back, sep="\n")

pretrain_back <- predict(backwards_holdout, train)
trainmse_back <- mse(train$V104, exp(pretrain_back))
cat("Backwards selection holdout train MSE is", trainmse_back, sep="\n")


# Backwards cross-validation MSE 
k <- 10
MSEB <- numeric(k)
set.seed(1)
folds <- sample(x=1:k, size=nrow(resdata), replace=TRUE)
for(j in 1:k){
  bcv_new = resdata[folds==j,]
  pred_b <- predict(backwards_holdout, newdata=bcv_new)
  MSEB[j] <- mse(bcv_new$V104, exp(pred_b))
}
cvmse_back <- weighted.mean(MSEB, table(folds)/sum(folds))
cat("Backwards selection Cross Validation MSE is: ", cvmse_back)

# another way to do cross-validation
library(caret)
set.seed(1)
train_cv <- trainControl(method="cv", number=10)
backwards_cv <- train(log(V104)~.-V105, data=resdata, method="lmStepAIC",
                      direction= "backward", trControl=train_cv, trace=FALSE)  #using CV generating another model


## Stepwise selection Linear Regression model
# ------------------------------------------------
start_time <- Sys.time()
# generating a maximum model hold out
max_model_holdout = lm(log(V104)~.-V105, data=train)
# fit a null model hold out, this model only includes intercept
null_model_holdout = lm(log(V104)~1, data=train)
# fit a minimal adequate model
stepwise_holdout = stepAIC(null_model_holdout, direction="both",
                           scope=list(upper=max_model_holdout, lower=null_model_holdout),trace=0)

summary(stepwise_holdout)
end_time <- Sys.time()
cat("Stepwise selection holdout computational time is", end_time-start_time, "seconds")


# test and train MSE
pretest_step <- predict(stepwise_holdout, test)
testmse_step <- mse(test$V104, exp(pretest_step))
cat("Stepwise selection holdout test MSE is", testmse_step, sep="\n")

pretrain_step <- predict(stepwise_holdout, train)
trainmse_step <- mse(train$V104, exp(pretrain_step))
cat("Backwards selection holdout train MSE is", trainmse_step, sep="\n")

# stepwise selection CV 
stepwise_cv <- train(null_model_holdout, data=resdata, method="lmStepAIC",
                     direction= "both", trace=FALSE,
                     scope=list(upper=max_model_holdout, lower=null_model_holdout),
                     trControl=train_cv)
summary(stepwise_cv)
#stepwise_cv1 <- train(log(V104)~.-V105, data=resdata, method="leapSeq",trControl=train_cv, trace=FALSE)   #error

# Stepwise selection CV MSE
MSES <- numeric(k)
for(j in 1:k){
  test = resdata[folds==j,]
  pred <- predict(stepwise_holdout, newdata=test)
  MSES[j] <- mse(test$V104, exp(pred))
}
MSES
cvmse_step <- weighted.mean(MSES, table(folds)/sum(folds))
cat("Stepwise selection Cross Validation MSE is", cvmse_step, sep="\n")


## Ridge regression
# ------------------------------------------------
#OLS
ols <- lm(log(V104)~.-V105, data=trainr)
summary(ols)
pre_ols <- predict(ols, newdata=testr)


lm.mod <- lm(yr[indexr]~Xr[indexr,])
summary(lm.mod)
lm.pred <- coef(lm.mod)[1] + Xr[-indexr,] %*% coef(lm.mod)[-1]
# above won't work, as it contains NA, must convert NA to 0 first
d <- coef(lm.mod)
d[is.na(d)] <-0
lm.pred <- d[1] + Xr[-indexr,] %*% d[-1]   # the same as pre_ols


library(glmnet)
set.seed(1)
gridr <- 10^seq(5, -2, length=100)

Xr <- model.matrix(log(V104)~.-V105, resdata)[,-1]
yr <- log(resdata$V104)

start_time <- Sys.time()
# Ridge parameter lambda tuning, indexr the same as to before
ridge <- glmnet(Xr[indexr,], yr[indexr], alpha = 0, lambda = gridr,thresh=1e-12)

#find the best lambda by cross-validation
set.seed(1)
ridge_cv <- cv.glmnet(Xr[indexr,], yr[indexr], alpha = 0,lambda=gridr,nflods=10,thresh=1e-12)
plot(ridge_cv)
plot(ridge_cv$glmnet.fit, "lambda", label=TRUE)    #this plot takes more time
 

bestlam_ridge <- ridge_cv$lambda.min
cat("Ridge regression optimal value of lambda is: ", bestlam_ridge, "\n")
end_time <- Sys.time()
cat("Ridge regression computational time is", end_time-start_time, "seconds")


# Ridge holdout MSE
#ridge <- glmnet(X[indexr,], y[indexr], alpha = 0, lambda = bestlam_ridge)
#pre_ridge <- predict(ridge, s = bestlam_ridge, newx = X[-indexr,]) values slightly different with below   
pre_ridge <- predict(ridge_cv, s = bestlam_ridge, newx = X[-indexr,])

testmse_ridge <- mse(exp(y[-indexr]), exp(pre_ridge))
cat("Ridge regression holdout MSE is", testmse_ridge, sep="\n")


# Ridge CV MSE
k <- 10
MSER <- numeric(k)
set.seed(1)
folds <- sample(x=1:k, size=nrow(resdata), replace=TRUE)
for(j in 1:k){
  pred_r <- predict(ridge_cv, s=bestlam_ridge, newx=Xr[folds==j,])
  MSER[j] <- mse(exp(yr[folds==j]), exp(pred_r))
}
cvmse_ridge <- weighted.mean(MSER, table(folds)/sum(folds))
cat("Ridge regression Cross Validation MSE is: ", cvmse_ridge)

#Look at the Ridge coefficients
ridge.coef <- predict(ridge, type = "coefficients", s = bestlam_ridge)
ridge.coef[,1][order(-abs(ridge.coef[,1]))]


## LASSO
# ------------------------------------------------
start_time <- Sys.time()
# Lasso parameter lambda tuning, indexr the same as to before
lasso <- glmnet(X[row.number,], y[row.number], alpha = 1, lambda =params)

#find the best lambda by cross-validation
set.seed(1)
lasso_cv <- cv.glmnet(X[row.number,], y[row.number], alpha = 1,lambda=params,nfolds=10,thresh=1e-12)
plot(lasso_cv$glmnet.fit, "lambda", label=TRUE)
plot(lasso_cv)

bestlam_lasso <- lasso_cv$lambda.min
cat("LASSO regression best lambda is", bestlam_lasso, sep="\n")
#lasso <- glmnet(X[row.number,], y[row.number], alpha = 1, lambda =bestlam_lasso)

end_time <- Sys.time()
cat("LASSO regression computational time is", end_time-start_time, "seconds")

# LASSO holdout MSE
pre_lasso <- predict(lasso_cv, s = bestlam_lasso, newx = Xr[-indexr,])
testmse_lasso <- mse(exp(yr[-indexr]), exp(pre_lasso))
cat("LASSO regression holdout MSE is: ", testmse_lasso)

# LASSO CV MSE
k <- 10
MSEL <- numeric(k)
set.seed(1)
folds <- sample(x=1:k, size=nrow(resdata), replace=TRUE)
for(j in 1:k){
  pred <- predict(lasso_cv, s=bestlam_lasso, newx=Xr[folds==j,])
  MSEL[j] <- mse(exp(yr[folds==j]), exp(pred))
}
cvmse_lasso <- weighted.mean(MSEL, table(folds)/sum(folds))
cat("LASSO regression Cross Validation MSE is: ", cvmse_lasso)

# LASSO coefficients
lasso.coef <- predict(lasso, type = "coefficients", s = bestlam_lasso)
lasso.coef[,1][order(-abs(lasso.coef[,1]))]


## Comparison of model predictions
# ------------------------------------------------
plot(y[-row.number], pre_ridge, ylim=c(4,11), col="red",pch=20,
     xlab="log(y_test)", ylab="predicted")
points(y[-row.number], pre_lasso, col="blue", pch=1)
points(y[-row.number], pretest_back, col="orange", pch=6)
points(y[-row.number], pretest_step, col="black", pch=3)
abline(0, 1)
legend(5,11, legend=c("Ridge","LASSO","Backwards","Stepwise"),
       col=c("red","blue","orange","black"),pch=c(20,1,6,3))

df <- data.frame(y[-row.number],pre_ridge, pre_lasso, pretest_back, pretest_step)

 


















