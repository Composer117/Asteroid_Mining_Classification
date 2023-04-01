library(Rdimtools)
library(randomForest)
library(caret)
library(gbm)


asteroid_df <- read.csv("Data/asteroid_dataset.csv")
asteroid_df$class_var <- factor(asteroid_df$class_var)
features <- colnames(asteroid_df)[-36]
asteroid_feature_matrix <- as.matrix(asteroid_df[features])

set.seed(31)

# Fischer Score Variables
do.fscore(asteroid_feature_matrix, as.vector(asteroid_df$class_var), ndim = 5)$featidx

# Mean Decrease Accuracy RF Importance top 5
rand_frst <- randomForest(asteroid_df$class_var ~ .,data = asteroid_df[features], 
                          ntree=100, keep.forest=FALSE,
                          importance=TRUE)
importance_df <- data.frame(index = 1:nrow(rand_frst$importance), rand_frst$importance)
sort(head(importance_df[order(importance_df$MeanDecreaseAccuracy, decreasing = TRUE), ]$index, 
          n=8))

# Mean Decrease Gini
sort(head(importance_df[order(importance_df$MeanDecreaseGini, decreasing = TRUE), ]$index, 
     n = 8))

# GBM Fit Model Variable Importance

gbmModel <- gbm(asteroid_df$class_var ~ ., data = asteroid_df[features], 
                    distribution = "gaussian", 
                    shrinkage = .01,
                    n.minobsinnode = 10,
                    n.trees = 200)

summary(gbmModel)

# SVM Fit Model Variable Importance

SVMmodel <- train(class_var ~ ., data = asteroid_df, 
               method = "svmRadial")
SVMImp <- varImp(SVMmodel)

head(sort(rowSums(SVMImp$importance), decreasing = TRUE), n = 8)

