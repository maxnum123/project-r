#load libraries
library("ggplot2")
library("e1071")
library(dplyr)
library(reshape2)
library(corrplot)
library(caret)
library(pROC)
library(gridExtra)
library(grid)
library(ggfortify)
library(purrr)
library(nnet)
library(doParallel) # parallel processing
registerDoParallel()
require(foreach)
require(iterators)
require(parallel)

#Loading raw Data set
Cancer.rawdata <- read.csv("C:/Users/Yael/Desktop/R project/Breast Cancer Wisconsin.csv", sep=",")

# Getting descriptive statistics
str(Cancer.rawdata)

#Remove the first  column
bc_data <- Cancer.rawdata[,-c(0:1)]
#Remove the last column
bc_data <- bc_data[,-32]
#Tidy the data
bc_data$diagnosis <- as.factor(bc_data$diagnosis)

head(bc_data)

#check for missing variables
sapply(bc_data, function(x) sum(is.na(x)))

summary(bc_data)

## Create a frequency table
diagnosis.table <- table(bc_data$diagnosis)
colors <- terrain.colors(2) 
# Create a pie chart 
diagnosis.prop.table <- prop.table(diagnosis.table)*100
diagnosis.prop.df <- as.data.frame(diagnosis.prop.table)
pielabels <- sprintf("%s - %3.1f%s", diagnosis.prop.df[,1], diagnosis.prop.table, "%")

pie(diagnosis.prop.table,
    labels=pielabels,  
    clockwise=TRUE,
    col=colors,
    border="gainsboro",
    radius=0.8,
    cex=0.8, 
    main="frequency of cancer diagnosis")
legend(1, .4, legend=diagnosis.prop.df[,1], cex = 0.7, fill = colors)

#Break up columns into groups, according to their suffix designation 
#(_mean, _se,and __worst) to perform visualisation plots off.
data_mean <- Cancer.rawdata[ ,c("diagnosis", "radius_mean", "texture_mean","perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave.points_mean", "symmetry_mean", "fractal_dimension_mean" )]

data_se <- Cancer.rawdata[ ,c("diagnosis", "radius_se", "texture_se","perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave.points_se", "symmetry_se", "fractal_dimension_se" )]

data_worst <- Cancer.rawdata[ ,c("diagnosis", "radius_worst", "texture_worst","perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave.points_worst", "symmetry_worst", "fractal_dimension_worst" )]

#Plot histograms of "_mean" variables group by diagnosis
ggplot(data = melt(data_mean, id.var = "diagnosis"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales =      'free_x')


#Plot histograms of "_se" variables group by diagnosis
ggplot(data = melt(data_se, id.var = "diagnosis"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales = 'free_x')

#Plot histograms of "_worst" variables group by diagnosis
ggplot(data = melt(data_worst, id.var = "diagnosis"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales = 'free_x')

# calculate collinearity
corMatMy <- cor(bc_data[,2:31])
corrplot(corMatMy, order = "hclust", tl.cex = 0.7)

highlyCor <- colnames(bc_data)[findCorrelation(corMatMy, cutoff = 0.9, verbose = TRUE)]

highlyCor

bc_data_cor <- bc_data[, which(!colnames(bc_data) %in% highlyCor)]
ncol(bc_data_cor)

cancer.pca <- prcomp(bc_data[, 2:31], center=TRUE, scale=TRUE)
plot(cancer.pca, type="l", main='')
grid(nx = 10, ny = 14)
title(main = "Principal components weight", sub = NULL, xlab = "Components")
box()

summary(cancer.pca)

# Calculate the proportion of variance explained
pca_var <- cancer.pca$sdev^2
pve_df <- pca_var / sum(pca_var)
cum_pve <- cumsum(pve_df)
pve_table <- tibble(comp = seq(1:ncol(bc_data %>% select(-diagnosis))), pve_df, cum_pve)

ggplot(pve_table, aes(x = comp, y = cum_pve)) + 
  geom_point() + 
  geom_abline(intercept = 0.95, color = "red", slope = 0)

cancer.pca2 <- prcomp(bc_data_cor, center=TRUE, scale=TRUE)
summary(cancer.pca2)

# Calculate the proportion of variance explained
pca_var2 <- cancer.pca2$sdev^2
pve_df2 <- pca_var2 / sum(pca_var2)
cum_pve2 <- cumsum(pve_df2)
pve_table2 <- tibble(comp = seq(1:ncol(bc_data_cor)), pve_df2, cum_pve2)

ggplot(pve_table2, aes(x = comp, y = cum_pve2)) + 
  geom_point() + 
  geom_abline(intercept = 0.95, color = "red", slope = 0)

pca_df <- as.data.frame(cancer.pca2$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col=bc_data$diagnosis)) + geom_point(alpha=0.5)

autoplot(cancer.pca2, data = bc_data,  colour = 'diagnosis',
         loadings = FALSE, loadings.label = TRUE, loadings.colour = "blue")

df_pcs <- cbind(as_tibble(bc_data$diagnosis), as_tibble(cancer.pca2$x))
GGally::ggpairs(df_pcs, columns = 2:4, ggplot2::aes(color = value))

#Split data set in train 70% and test 30%
set.seed(1234)
df <- cbind(diagnosis = bc_data$diagnosis, bc_data_cor)
train_indx <- createDataPartition(df$diagnosis, p = 0.7, list = FALSE)

train_set <- df[train_indx,]
test_set <- df[-train_indx,]

nrow(train_set)

nrow(test_set)

fitControl <- trainControl(method="cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

model_rf <- train(diagnosis~.,
                  data = train_set,
                  method="rf",
                  metric="ROC",
                  #tuneLength=10,
                  preProcess = c('center', 'scale'),
                  trControl=fitControl)

# plot feature importance
plot(varImp(model_rf), top = 10, main = "Random forest")

pred_rf <- predict(model_rf, test_set)
cm_rf <- confusionMatrix(pred_rf, test_set$diagnosis, positive = "M")
cm_rf

model_pca_rf <- train(diagnosis~.,
                      data = train_set,
                      method="ranger",
                      metric="ROC",
                      #tuneLength=10,
                      preProcess = c('center', 'scale', 'pca'),
                      trControl=fitControl)

pred_pca_rf <- predict(model_pca_rf, test_set)
cm_pca_rf <- confusionMatrix(pred_pca_rf, test_set$diagnosis, positive = "M")
cm_pca_rf

model_knn <- train(diagnosis~.,
                   data = train_set,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10,
                   trControl=fitControl)

pred_knn <- predict(model_knn, test_set)
cm_knn <- confusionMatrix(pred_knn, test_set$diagnosis, positive = "M")
cm_knn

model_nnet <- train(diagnosis~.,
                    data = train_set,
                    method="nnet",
                    metric="ROC",
                    preProcess=c('center', 'scale'),
                    trace=FALSE,
                    tuneLength=10,
                    trControl=fitControl)

pred_nnet <- predict(model_nnet, test_set)
cm_nnet <- confusionMatrix(pred_nnet, test_set$diagnosis, positive = "M")
cm_nnet

model_pca_nnet <- train(diagnosis~.,
                        data = train_set,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)

pred_pca_nnet <- predict(model_pca_nnet, test_set)
cm_pca_nnet <- confusionMatrix(pred_pca_nnet, test_set$diagnosis, positive = "M")
cm_pca_nnet

model_svm <- train(diagnosis~.,
                   data = train_set,
                   method="svmRadial",
                   metric="ROC",
                   preProcess=c('center', 'scale'),
                   trace=FALSE,
                   trControl=fitControl)

pred_svm <- predict(model_svm, test_set)
cm_svm <- confusionMatrix(pred_svm, test_set$diagnosis, positive = "M")
cm_svm

model_nb <- train(diagnosis~.,
                  data = train_set,
                  method="nb",
                  metric="ROC",
                  preProcess=c('center', 'scale'),
                  trace=FALSE,
                  trControl=fitControl)

pred_nb <- predict(model_nb, test_set)
cm_nb <- confusionMatrix(pred_nb, test_set$diagnosis, positive = "M")
cm_nb

model_list <- list(RF=model_rf, PCA_RF=model_pca_rf, 
                   NNET=model_nnet, PCA_NNET=model_pca_nnet,  
                   KNN = model_knn, SVM=model_svm, NB=model_nb)
resamples <- resamples(model_list)

bwplot(resamples, metric = "ROC")

cm_list <- list(RF=cm_rf, PCA_RF=cm_pca_rf, 
                NNET=cm_nnet, PCA_NNET=cm_pca_nnet,  
                KNN = cm_knn, SVM=cm_svm, NB=cm_nb)

results <- sapply(cm_list, function(x) x$byClass)
results

results_max <- apply(results, 1, which.is.max)

output_report <- data.frame(metric=names(results_max), 
                            best_model=colnames(results)[results_max],
                            value=mapply(function(x,y) {results[x,y]}, 
                                         names(results_max), 
                                         results_max))
rownames(output_report) <- NULL
output_report


