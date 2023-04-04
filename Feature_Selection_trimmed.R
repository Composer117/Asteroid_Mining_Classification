library(Rdimtools)
library(randomForest)
library(caret)
library(gbm)


asteroid_df <- read.csv("Data/asteroid_dataset.csv")

sort(table(asteroid_df$class_var))
frequency <- data.frame(table(asteroid_df$class_var))
trimmed_frequency <- frequency[frequency$Freq >10, "Var1"]

asteroid_df_trimmed <- asteroid_df[asteroid_df$class_var %in% trimmed_frequency,]

asteroid_df_trimmed$class_var <- factor(asteroid_df_trimmed$class_var)
features <- colnames(asteroid_df_trimmed)[-36]
asteroid_feature_matrix <- as.matrix(asteroid_df_trimmed[features])

set.seed(1)

# Fisher Score Variables
Fisher_index <- do.fscore(asteroid_feature_matrix, as.vector(asteroid_df_trimmed$class_var), ndim = 10)$featidx
features[Fisher_index]


# Mean Decrease Accuracy RF Importance
rand_frst <- randomForest(asteroid_df_trimmed$class_var ~ ., data = asteroid_df_trimmed[features], 
                          ntree=100, keep.forest=FALSE,
                          importance=TRUE)
importance_df <- data.frame(index = 1:nrow(rand_frst$importance), rand_frst$importance)
RF_acc_index <- sort(head(importance_df[order(importance_df$MeanDecreaseAccuracy, decreasing = TRUE), ]$index, 
          n=10))
features[RF_acc_index]


# Mean Decrease Gini
RF_gini_index <- sort(head(importance_df[order(importance_df$MeanDecreaseGini, decreasing = TRUE), ]$index, 
     n = 10))
features[RF_gini_index]

# GBM Fit Model Variable Importance

gbmModel <- gbm(class_var ~ ., data = asteroid_df_trimmed, 
                    distribution = "gaussian", 
                    shrinkage = .01,
                    n.minobsinnode = 10,
                    n.trees = 200)

head(summary(gbmModel), n=10)$var

# SVM Fit Model Variable Importance

SVMmodel <- train(class_var ~ ., data = asteroid_df_trimmed, 
               method = "svmRadial")
SVMImp <- varImp(SVMmodel)

names(head(sort(rowSums(SVMImp$importance), decreasing = TRUE), n = 10))

# Exporting trimmed dataframe to csv

write.csv(asteroid_df_trimmed, "Data/asteroid_dataset_trimmed.csv")

