library(caret)
library(rsample)
library(MASS)
library(yardstick)

# Creating function to calculate MCC by class

calculate_mcc_by_class <- function(cm) {
  num_classes <- dim(cm$table)[1]
  mcc_by_class <- numeric(num_classes)
  for (i in 1:num_classes) {
    tp <- cm$table[i,i]
    fp <- sum(cm$table[,i]) - tp
    fn <- sum(cm$table[i,]) - tp
    tn <- sum(cm$table) - tp - fp - fn
    mcc_by_class[i] <- (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  }
  names(mcc_by_class) <- rownames(cm$table)
  return(mcc_by_class)
}

calculate_weighted_average_multiclass <- function(actual, predicted) {
  # calculate confusion matrix using caret
  cm <- confusionMatrix(predicted, actual, mode = "everything")
  
  # calculate class distribution of actual data
  class_counts <- table(actual)
  class_dist <- class_counts / sum(class_counts)
  
  # extract TP, FP, precision, recall, and F-measure for each class
  tp <- cm$byClass[, "Sensitivity"]
  fp <- cm$byClass[, "Specificity"]
  precision <- cm$byClass[, "Precision"]
  recall <- cm$byClass[, "Sensitivity"]
  f_measure <- cm$byClass[, "F1"]
  
  # calculate weighted average
  weighted_tp_rate <- sum(tp * class_dist)
  weighted_fp_rate <- sum(fp * class_dist)
  weighted_precision <- sum(precision * class_dist)
  weighted_recall <- sum(recall * class_dist)
  weighted_f_measure <- sum(f_measure * class_dist)
  
  # return results as a list
  return(list(
    TP_rate = weighted_tp_rate,
    FP_rate = weighted_fp_rate,
    Precision = weighted_precision,
    Recall = weighted_recall,
    F_measure = weighted_f_measure
  ))
}


asteroid_df <- read.csv("Data/asteroid_dataset_trimmed.csv")
asteroid_df$class_var <- factor(asteroid_df$class_var)
features <- colnames(asteroid_df)[-36]

# The classification algorithms that we are running for this 
# use case are as follows:
# 1. Decision Tree - RPart
# 2. Random Forest
# 3. NNet
# 4. Bagged Adaboost
# 5. SVM - Radial

set.seed(31)


# Data Preparation

split <- initial_split(asteroid_df, prop = 0.66, strata = class_var)
train <- training(split)
test <- testing(split)

write.csv(train, "Data/asteroid_train.csv")
write.csv(test, "Data/asteroid_test.csv")

## First, we will run all algorithms on all variables
# Starting with rpart

train_control <- trainControl(method = "repeatedcv", number = 10, 
                              summaryFunction = defaultSummary)

model_rpart <- train(class_var ~ ., data = train, 
                     method = "rpart", trControl = train_control,
                     tuneLength = 10)

test_pred_rpart <- predict(model_rpart, newdata = test)
test_pred_rpart
confusionMatrix(test_pred_rpart, test$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_rpart, test$class_var, mode = "everything"))
mcc(preds = test_pred_rpart, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_rpart)



# Random Forest

train_control_RF <- trainControl(method = "CV", 
                                  number = 10,
                                  summaryFunction = defaultSummary, 
                                 classProbs = TRUE,
                                 savePredictions = TRUE)
mtryValues <- seq(2, ncol(asteroid_df)-1, by = 1)
rfFit <- train(x = train[features], 
               y = train$class_var,
               method = "rf",
               ntree = 100,
               tuneGrid = data.frame(mtry = mtryValues),
               importance = TRUE,
               metric = "ROC",
               trControl = train_control_RF)

pred_RF <- predict(rfFit, test)
confusionMatrix(pred_RF, test$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(pred_RF, test$class_var, mode = "everything"))
mcc(preds = pred_RF, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, pred_RF)

# NNet

train_control_Nnet <- trainControl(method = "CV", number = 10,
                              classProbs = TRUE, 
                              summaryFunction = defaultSummary,
                              savePredictions = TRUE)

nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1, 2))

nnetFit <- train(x = train[features], 
                 y = train$class_var,
                 method = "nnet",
                 metric = "ROC",
                 preProc = c("center", "scale"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 100,
                 MaxNWts = 1000,
                 trControl = train_control_Nnet)

test_pred_nnet <- predict(nnetFit, newdata = test)
confusionMatrix(test_pred_nnet, test$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_nnet, test$class_var, mode = "everything"))
mcc(preds = test_pred_nnet, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_nnet)


# Bagged AdaBoost

train_control_SVM <- trainControl(method = "repeatedcv", 
                                  number = 10,
                                  summaryFunction = defaultSummary)

model_adaboost <- train(class_var ~ ., data = train, method = "AdaBag",
                  trControl = train_control_SVM)

test_pred_adaboost <- predict(model_adaboost, newdata = test)
confusionMatrix(test_pred_adaboost, test$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_adaboost, test$class_var, mode = "everything"))
mcc(preds = test_pred_adaboost, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_adaboost)

# SVM-radial

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, 
                              summaryFunction = defaultSummary)
svmGrid <-  expand.grid(sigma = 0.1, C = 1.0)

SVM_model <- train(class_var ~ ., data = asteroid_df, method = "svmRadial",
               preProc = c("center", "scale"),
               trControl = train_control, tuneGrid = svmGrid)
test_pred_svm <- predict(SVM_model, newdata = test)
confusionMatrix(test_pred_svm, test$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_svm, test$class_var, mode = "everything"))
mcc(preds = test_pred_svm, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_svm)




##### Running all algorithms with top 10 variables according to - Fisher Score #####

features_fisher <- c("albedo", "diameter", "per", "a", "ad", "moid_jup", 
                     "moid", "sigma_per", "q", "sigma_a", "class_var")

train_fisher <- train[features_fisher]
test_fisher <- test[features_fisher]

# RPart

model_rpart_fisher <- train(class_var ~ ., data = train_fisher, 
                     method = "rpart", trControl = train_control,
                     tuneLength = 10)

test_pred_rpart_fisher <- predict(model_rpart_fisher, newdata = test_fisher)
confusionMatrix(test_pred_rpart_fisher, test_fisher$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_rpart_fisher, test$class_var, mode = "everything"))
mcc(preds = test_pred_rpart_fisher, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_rpart_fisher)

# Random Forest

mtryValues_f <- seq(2, ncol(train_fisher)-1, by = 1)
rfFit_fisher <- train(x = train_fisher[-9], 
               y = train_fisher$class_var,
               method = "rf",
               ntree = 100,
               tuneGrid = data.frame(mtry = mtryValues_f),
               importance = TRUE,
               metric = "ROC",
               trControl = train_control_RF)

pred_RF_fisher <- predict(rfFit_fisher, test_fisher)
confusionMatrix(pred_RF_fisher, test_fisher$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(pred_RF_fisher, test$class_var, mode = "everything"))
mcc(preds = pred_RF_fisher, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, pred_RF_fisher)

# NNet

nnetFit_fisher <- train(x = train_fisher[-9], 
                 y = train_fisher$class_var,
                 method = "nnet",
                 metric = "ROC",
                 preProc = c("center", "scale"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 100,
                 MaxNWts = 1000,
                 trControl = train_control_Nnet)

test_pred_nnet_fisher <- predict(nnetFit_fisher, newdata = test_fisher)
confusionMatrix(test_pred_nnet_fisher, test_fisher$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_nnet_fisher, test$class_var, mode = "everything"))
mcc(preds = test_pred_nnet_fisher, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_nnet_fisher)

# SVM Radial

SVM_model_fisher <- train(class_var ~ ., data = train_fisher, method = "svmRadial",
                   preProc = c("center", "scale"),
                   trControl = train_control, tuneGrid = svmGrid)
test_pred_svm_fisher <- predict(SVM_model_fisher, newdata = test_fisher)
confusionMatrix(test_pred_svm_fisher, test_fisher$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_svm_fisher, test$class_var, mode = "everything"))
mcc(preds = test_pred_svm_fisher, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_svm_fisher)

# Bagged ADAboost

model_adaboost_fisher <- train(class_var ~ ., data = train_fisher, method = "AdaBag",
                        trControl = train_control_SVM)

test_pred_adaboost_fisher <- predict(model_adaboost_fisher, newdata = test_fisher)
confusionMatrix(test_pred_adaboost_fisher, test_fisher$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_adaboost_fisher, test$class_var, mode = "everything"))
mcc(preds = test_pred_adaboost_fisher, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_adaboost_fisher)


##### Running all algorithms with top 10 variables according to - RF-accuracy #####

features_rfacc <- c("diameter", "albedo", "a", "q", "n", "per", 
                    "moid", "t_jup", "sigma_a", "class_var")

train_rfacc <- train[features_rfacc]
test_rfacc <- test[features_rfacc]

# RPart

model_rpart_rfacc <- train(class_var ~ ., data = train_rfacc, 
                            method = "rpart", trControl = train_control,
                            tuneLength = 10)

test_pred_rpart_rfacc <- predict(model_rpart_rfacc, newdata = test_rfacc)
confusionMatrix(test_pred_rpart_rfacc, test_rfacc$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_rpart_rfacc, test$class_var, mode = "everything"))
mcc(preds = test_pred_rpart_rfacc, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_rpart_rfacc)

# Random Forest

mtryValues_rfacc <- seq(2, ncol(train_rfacc)-1, by = 1)
rfFit_rfacc <- train(x = train_rfacc[-9], 
                      y = train_rfacc$class_var,
                      method = "rf",
                      ntree = 100,
                      tuneGrid = data.frame(mtry = mtryValues_rfacc),
                      importance = TRUE,
                      metric = "ROC",
                      trControl = train_control_RF)

pred_RF_rfacc <- predict(rfFit_rfacc, test_rfacc)
confusionMatrix(pred_RF_rfacc, test_rfacc$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(pred_RF_rfacc, test$class_var, mode = "everything"))
mcc(preds = pred_RF_rfacc, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, pred_RF_rfacc)

# NNet

nnetFit_rfacc <- train(x = train_rfacc[-9], 
                        y = train_rfacc$class_var,
                        method = "nnet",
                        metric = "ROC",
                        preProc = c("center", "scale"),
                        tuneGrid = nnetGrid,
                        trace = FALSE,
                        maxit = 100,
                        MaxNWts = 1000,
                        trControl = train_control_Nnet)

test_pred_nnet_rfacc <- predict(nnetFit_rfacc, newdata = test_rfacc)
confusionMatrix(test_pred_nnet_rfacc, test_rfacc$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_nnet_rfacc, test$class_var, mode = "everything"))
mcc(preds = test_pred_nnet_rfacc, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_nnet_rfacc)

# SVM Radial

SVM_model_rfacc <- train(class_var ~ ., data = train_rfacc, method = "svmRadial",
                          preProc = c("center", "scale"),
                          trControl = train_control, tuneGrid = svmGrid)
test_pred_svm_rfacc <- predict(SVM_model_rfacc, newdata = test_rfacc)
confusionMatrix(test_pred_svm_rfacc, test_rfacc$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_svm_rfacc, test$class_var, mode = "everything"))
mcc(preds = test_pred_svm_rfacc, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_svm_rfacc)


# Bagged ADAboost

model_adaboost_rfacc <- train(class_var ~ ., data = train_rfacc, method = "AdaBag",
                               trControl = train_control_SVM)

test_pred_adaboost_rfacc <- predict(model_adaboost_rfacc, newdata = test_rfacc)
confusionMatrix(test_pred_adaboost_rfacc, test_rfacc$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_adaboost_rfacc, test$class_var, mode = "everything"))
mcc(preds = test_pred_adaboost_rfacc, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_adaboost_rfacc)


##### Running all algorithms with top 10 variables according to - RF-Gini #####

features_rfgini <- c("H", "diameter", "albedo", "a", "ad", "n", "per",
                     "moid_jup", "t_jup", "sigma_per", "class_var")

train_rfgini <- train[features_rfgini]
test_rfgini <- test[features_rfgini]

# RPart

model_rpart_rfgini <- train(class_var ~ ., data = train_rfgini, 
                           method = "rpart", trControl = train_control,
                           tuneLength = 10)

test_pred_rpart_rfgini <- predict(model_rpart_rfgini, newdata = test_rfgini)
confusionMatrix(test_pred_rpart_rfgini, test_rfgini$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_rpart_rfgini, test$class_var, mode = "everything"))
mcc(preds = test_pred_rpart_rfgini, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_rpart_rfgini)

# Random Forest

mtryValues_rfgini <- seq(2, ncol(train_rfgini)-1, by = 1)
rfFit_rfgini <- train(x = train_rfgini[-9], 
                     y = train_rfgini$class_var,
                     method = "rf",
                     ntree = 100,
                     tuneGrid = data.frame(mtry = mtryValues_rfgini),
                     importance = TRUE,
                     metric = "ROC",
                     trControl = train_control_RF)

pred_RF_rfgini <- predict(rfFit_rfgini, test_rfgini)
confusionMatrix(pred_RF_rfgini, test_rfgini$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(pred_RF_rfgini, test$class_var, mode = "everything"))
mcc(preds = pred_RF_rfgini, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, pred_RF_rfgini)

# NNet

nnetFit_rfgini <- train(x = train_rfgini[-9], 
                       y = train_rfgini$class_var,
                       method = "nnet",
                       metric = "ROC",
                       preProc = c("center", "scale"),
                       tuneGrid = nnetGrid,
                       trace = FALSE,
                       maxit = 100,
                       MaxNWts = 1000,
                       trControl = train_control_Nnet)

test_pred_nnet_rfgini <- predict(nnetFit_rfgini, newdata = test_rfgini)
confusionMatrix(test_pred_nnet_rfgini, test_rfgini$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_nnet_rfgini, test$class_var, mode = "everything"))
mcc(preds = test_pred_nnet_rfgini, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_nnet_rfgini)

# SVM Radial

SVM_model_rfgini <- train(class_var ~ ., data = train_rfgini, method = "svmRadial",
                         preProc = c("center", "scale"),
                         trControl = train_control, tuneGrid = svmGrid)
test_pred_svm_rfgini <- predict(SVM_model_rfgini, newdata = test_rfgini)
confusionMatrix(test_pred_svm_rfgini, test_rfgini$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_svm_rfgini, test$class_var, mode = "everything"))
mcc(preds = test_pred_svm_rfgini, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_svm_rfgini)

# Bagged ADAboost

model_adaboost_rfgini <- train(class_var ~ ., data = train_rfgini, method = "AdaBag",
                              trControl = train_control_SVM)

test_pred_adaboost_rfgini <- predict(model_adaboost_rfgini, newdata = test_rfgini)
confusionMatrix(test_pred_adaboost_rfgini, test_rfgini$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_adaboost_rfgini, test$class_var, mode = "everything"))
mcc(preds = test_pred_adaboost_rfgini, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_adaboost_rfgini)


##### Running all algorithms with top 10 variables according to - GBM Variable Importance #####

features_gbmVI <- c("albedo", "moid_jup", "a", "sigma_ma", "n", "sigma_tp", "i",
                    "rms",  "diameter","t_jup", "class_var")

train_gbmVI <- train[features_gbmVI]
test_gbmVI <- test[features_gbmVI]

# RPart

model_rpart_gbmVI <- train(class_var ~ ., data = train_gbmVI, 
                            method = "rpart", trControl = train_control,
                            tuneLength = 10)

test_pred_rpart_gbmVI <- predict(model_rpart_gbmVI, newdata = test_gbmVI)
confusionMatrix(test_pred_rpart_gbmVI, test_gbmVI$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_rpart_gbmVI, test$class_var, mode = "everything"))
mcc(preds = test_pred_rpart_gbmVI, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_rpart_gbmVI)

# Random Forest

mtryValues_gbmVI <- seq(2, ncol(train_gbmVI)-1, by = 1)
rfFit_gbmVI <- train(x = train_gbmVI[-9], 
                      y = train_gbmVI$class_var,
                      method = "rf",
                      ntree = 100,
                      tuneGrid = data.frame(mtry = mtryValues_gbmVI),
                      importance = TRUE,
                      metric = "ROC",
                      trControl = train_control_RF)

pred_RF_gbmVI <- predict(rfFit_gbmVI, test_gbmVI)
confusionMatrix(pred_RF_gbmVI, test_gbmVI$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(pred_RF_gbmVI, test$class_var, mode = "everything"))
mcc(preds = pred_RF_gbmVI, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, pred_RF_gbmVI)

# NNet

nnetFit_gbmVI <- train(x = train_gbmVI[-9], 
                        y = train_gbmVI$class_var,
                        method = "nnet",
                        metric = "ROC",
                        preProc = c("center", "scale"),
                        tuneGrid = nnetGrid,
                        trace = FALSE,
                        maxit = 100,
                        MaxNWts = 1000,
                        trControl = train_control_Nnet)

test_pred_nnet_gbmVI <- predict(nnetFit_gbmVI, newdata = test_gbmVI)
confusionMatrix(test_pred_nnet_gbmVI, test_gbmVI$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_nnet_gbmVI, test$class_var, mode = "everything"))
mcc(preds = test_pred_nnet_gbmVI, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_nnet_gbmVI)

# SVM Radial

SVM_model_gbmVI <- train(class_var ~ ., data = train_gbmVI, method = "svmRadial",
                         preProc = c("center", "scale"),
                         trControl = train_control, tuneGrid = svmGrid)
test_pred_svm_gbmVI <- predict(SVM_model_gbmVI, newdata = test_gbmVI)
confusionMatrix(test_pred_svm_gbmVI, test_gbmVI$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_svm_gbmVI, test$class_var, mode = "everything"))
mcc(preds = test_pred_svm_gbmVI, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_svm_gbmVI)

# Bagged ADAboost

model_adaboost_gbmVI <- train(class_var ~ ., data = train_gbmVI, method = "AdaBag",
                               trControl = train_control_SVM)

test_pred_adaboost_gbmVI <- predict(model_adaboost_gbmVI, newdata = test_gbmVI)
confusionMatrix(test_pred_adaboost_gbmVI, test_gbmVI$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_adaboost_gbmVI, test$class_var, mode = "everything"))
mcc(preds = test_pred_adaboost_gbmVI, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_adaboost_gbmVI)



##### Running all algorithms with top 10 variables according to - SVM Variable Importance #####

features_svmVI <- c("albedo", "diameter", "rms", "ad", "moid_jup", "sigma_ma", 
                    "a", "n", "per", "t_jup", "class_var")

train_svmVI <- train[features_svmVI]
test_svmVI <- test[features_svmVI]

# RPart

model_rpart_svmVI <- train(class_var ~ ., data = train_svmVI, 
                           method = "rpart", trControl = train_control,
                           tuneLength = 10)

test_pred_rpart_svmVI <- predict(model_rpart_svmVI, newdata = test_svmVI)
confusionMatrix(test_pred_rpart_svmVI, test_svmVI$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_rpart_svmVI, test$class_var, mode = "everything"))
mcc(preds = test_pred_rpart_svmVI, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_rpart_svmVI)

# Random Forest

mtryValues_svmVI <- seq(2, ncol(train_svmVI)-1, by = 1)
rfFit_svmVI <- train(x = train_svmVI[-9], 
                     y = train_svmVI$class_var,
                     method = "rf",
                     ntree = 100,
                     tuneGrid = data.frame(mtry = mtryValues_svmVI),
                     importance = TRUE,
                     metric = "ROC",
                     trControl = train_control_RF)

pred_RF_svmVI <- predict(rfFit_svmVI, test_svmVI)
confusionMatrix(pred_RF_svmVI, test_svmVI$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(pred_RF_svmVI, test$class_var, mode = "everything"))
mcc(preds = pred_RF_svmVI, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, pred_RF_svmVI)

# NNet

nnetFit_svmVI <- train(x = train_svmVI[-9], 
                       y = train_svmVI$class_var,
                       method = "nnet",
                       metric = "ROC",
                       preProc = c("center", "scale"),
                       tuneGrid = nnetGrid,
                       trace = FALSE,
                       maxit = 100,
                       MaxNWts = 1000,
                       trControl = train_control_Nnet)

test_pred_nnet_svmVI <- predict(nnetFit_svmVI, newdata = test_svmVI)
confusionMatrix(test_pred_nnet_svmVI, test_svmVI$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_nnet_svmVI, test$class_var, mode = "everything"))
mcc(preds = test_pred_nnet_svmVI, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_nnet_svmVI)

# SVM Radial

SVM_model_svmVI <- train(class_var ~ ., data = train_svmVI, method = "svmRadial",
                         preProc = c("center", "scale"),
                         trControl = train_control, tuneGrid = svmGrid)
test_pred_svm_svmVI <- predict(SVM_model_svmVI, newdata = test_svmVI)
confusionMatrix(test_pred_svm_svmVI, test_svmVI$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_svm_svmVI, test$class_var, mode = "everything"))
mcc(preds = test_pred_svm_svmVI, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_svm_svmVI)

# Bagged ADAboost

model_adaboost_svmVI <- train(class_var ~ ., data = train_svmVI, method = "AdaBag",
                              trControl = train_control_SVM)

test_pred_adaboost_svmVI <- predict(model_adaboost_svmVI, newdata = test_svmVI)
confusionMatrix(test_pred_adaboost_svmVI, test_svmVI$class_var, mode = "everything")
calculate_mcc_by_class(confusionMatrix(test_pred_adaboost_svmVI, test$class_var, mode = "everything"))
mcc(preds = test_pred_adaboost_svmVI, actuals = test$class_var)
calculate_weighted_average_multiclass(test$class_var, test_pred_adaboost_svmVI)


# Best Model is the RF Accuracy model
# Saving the dataset into seperate CSV files

write.csv(train_rfacc, "Data/asteroid_train_rfacc.csv")
write.csv(test_rfacc, "Data/asteroid_test_rfacc.csv")






