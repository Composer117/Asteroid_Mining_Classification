library(caret)
library(rsample)
library(MASS)


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

# First, we will run all algorithms on all variables


split <- initial_split(asteroid_df, prop = 0.66, strata = class_var)
train <- training(split)
test <- testing(split)

# Starting with rpart

train_control <- trainControl(method = "repeatedcv", number = 10, 
                              summaryFunction = defaultSummary)

model_rpart <- train(class_var ~ ., data = train, 
                     method = "rpart", trControl = train_control,
                     tuneLength = 10)

test_pred_rpart <- predict(model_rpart, newdata = test)
confusionMatrix(test_pred_rpart, test$class_var, mode = "everything")



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

# Bagged CART

#train_control <- trainControl(method = "repeatedcv", number = 10, 
#                              summaryFunction = defaultSummary)

#model_treebag <- train(class_var ~ ., data = train, method = "treebag", 
#                   trControl = train_control,
#                   tuneLength = 4)

#test_pred_treebag <- predict(model_treebag, newdata = test)
#confusionMatrix(test_pred_treebag, test$class_var)

# Bagged AdaBoost

train_control_SVM <- trainControl(method = "repeatedcv", 
                                  number = 10,
                                  summaryFunction = defaultSummary)

model_adaboost <- train(class_var ~ ., data = train, method = "AdaBag",
                  trControl = train_control_SVM)

test_pred_adaboost <- predict(model_adaboost, newdata = test)
confusionMatrix(test_pred_adaboost, test$class_var, mode = "everything")

# SVM-radial

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, 
                              summaryFunction = defaultSummary)
svmGrid <-  expand.grid(sigma = 0.1, C = 1.0)

SVM_model <- train(class_var ~ ., data = asteroid_df, method = "svmRadial",
               preProc = c("center", "scale"),
               trControl = train_control, tuneGrid = svmGrid)
test_pred_svm <- predict(SVM_model, newdata = test)
confusionMatrix(test_pred_svm, test$class_var, mode = "everything")




##### Running all algorithms with top 8 variables according to - Fisher Score #####

features_fisher <- c("diameter", "tp", "e",  "ma",   "moid",  "H", "per", "sigma_tp",
                     "a","sigma_e", "class_var")

train_fisher <- train[features_fisher]
test_fisher <- test[features_fisher]

# RPart

model_rpart_fisher <- train(class_var ~ ., data = train_fisher, 
                     method = "rpart", trControl = train_control,
                     tuneLength = 10)

test_pred_rpart_fisher <- predict(model_rpart_fisher, newdata = test_fisher)
confusionMatrix(test_pred_rpart_fisher, test_fisher$class_var, mode = "everything")

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

# SVM Radial

SVM_model_fisher <- train(class_var ~ ., data = train_fisher, method = "svmRadial",
                   preProc = c("center", "scale"),
                   trControl = train_control, tuneGrid = svmGrid)
test_pred_svm_fisher <- predict(SVM_model_fisher, newdata = test_fisher)
confusionMatrix(test_pred_svm_fisher, test_fisher$class_var, mode = "everything")

# Bagged ADAboost

model_adaboost_fisher <- train(class_var ~ ., data = train_fisher, method = "AdaBag",
                        trControl = train_control_SVM)

test_pred_adaboost_fisher <- predict(model_adaboost_fisher, newdata = test_fisher)
confusionMatrix(test_pred_adaboost_fisher, test_fisher$class_var, mode = "everything")


##### Running all algorithms with top 8 variables according to - RF-accuracy #####

features_rfacc <- c("H",        "diameter", "e",        "ad",       "tp",       "per",      "moid",     "moid_jup",
                    "sigma_e",  "sigma_tp", "class_var")

train_rfacc <- train[features_rfacc]
test_rfacc <- test[features_rfacc]

# RPart

model_rpart_rfacc <- train(class_var ~ ., data = train_rfacc, 
                            method = "rpart", trControl = train_control,
                            tuneLength = 10)

test_pred_rpart_rfacc <- predict(model_rpart_rfacc, newdata = test_rfacc)
confusionMatrix(test_pred_rpart_rfacc, test_rfacc$class_var, mode = "everything")

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

# SVM Radial

SVM_model_rfacc <- train(class_var ~ ., data = train_rfacc, method = "svmRadial",
                          preProc = c("center", "scale"),
                          trControl = train_control, tuneGrid = svmGrid)
test_pred_svm_rfacc <- predict(SVM_model_rfacc, newdata = test_rfacc)
confusionMatrix(test_pred_svm_rfacc, test_rfacc$class_var, mode = "everything")


# Bagged ADAboost

model_adaboost_rfacc <- train(class_var ~ ., data = train_rfacc, method = "AdaBag",
                               trControl = train_control_SVM)

test_pred_adaboost_rfacc <- predict(model_adaboost_rfacc, newdata = test_rfacc)
confusionMatrix(test_pred_adaboost_rfacc, test_rfacc$class_var, mode = "everything")


##### Running all algorithms with top 8 variables according to - RF-Gini #####

features_rfgini <- c("X",        "H",        "diameter", "albedo",   "e",        "ma",       "ad",       "tp",      
                     "moid",     "moid_jup", "class_var")

train_rfgini <- train[features_rfgini]
test_rfgini <- test[features_rfgini]

# RPart

model_rpart_rfgini <- train(class_var ~ ., data = train_rfgini, 
                           method = "rpart", trControl = train_control,
                           tuneLength = 10)

test_pred_rpart_rfgini <- predict(model_rpart_rfgini, newdata = test_rfgini)
confusionMatrix(test_pred_rpart_rfgini, test_rfgini$class_var, mode = "everything")

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

# SVM Radial

SVM_model_rfgini <- train(class_var ~ ., data = train_rfgini, method = "svmRadial",
                         preProc = c("center", "scale"),
                         trControl = train_control, tuneGrid = svmGrid)
test_pred_svm_rfgini <- predict(SVM_model_rfgini, newdata = test_rfgini)
confusionMatrix(test_pred_svm_rfgini, test_rfgini$class_var, mode = "everything")

# Bagged ADAboost

model_adaboost_rfgini <- train(class_var ~ ., data = train_rfgini, method = "AdaBag",
                              trControl = train_control_SVM)

test_pred_adaboost_rfgini <- predict(model_adaboost_rfgini, newdata = test_rfgini)
confusionMatrix(test_pred_adaboost_rfgini, test_rfgini$class_var, mode = "everything")


##### Running all algorithms with top 8 variables according to - GBM Variable Importance #####

features_gbmVI <- c("albedo",   "moid_jup", "sigma_tp", "n",        "a",        "diameter", "i",        "w",       
                    "rms",      "t_jup", "class_var")

train_gbmVI <- train[features_gbmVI]
test_gbmVI <- test[features_gbmVI]

# RPart

model_rpart_gbmVI <- train(class_var ~ ., data = train_gbmVI, 
                            method = "rpart", trControl = train_control,
                            tuneLength = 10)

test_pred_rpart_gbmVI <- predict(model_rpart_gbmVI, newdata = test_gbmVI)
confusionMatrix(test_pred_rpart_gbmVI, test_gbmVI$class_var, mode = "everything")

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

# SVM Radial

SVM_model_gbmVI <- train(class_var ~ ., data = train_gbmVI, method = "svmRadial",
                         preProc = c("center", "scale"),
                         trControl = train_control, tuneGrid = svmGrid)
test_pred_svm_gbmVI <- predict(SVM_model_gbmVI, newdata = test_gbmVI)
confusionMatrix(test_pred_svm_gbmVI, test_gbmVI$class_var, mode = "everything")

# Bagged ADAboost

model_adaboost_gbmVI <- train(class_var ~ ., data = train_gbmVI, method = "AdaBag",
                               trControl = train_control_SVM)

test_pred_adaboost_gbmVI <- predict(model_adaboost_gbmVI, newdata = test_gbmVI)
confusionMatrix(test_pred_adaboost_gbmVI, test_gbmVI$class_var, mode = "everything")



##### Running all algorithms with top 8 variables according to - SVM Variable Importance #####

features_svmVI <- c("albedo",   "diameter", "rms",      "ad",       "moid_jup", "sigma_ma", "a",        "n",       
                    "per",      "t_jup", "class_var")

train_svmVI <- train[features_svmVI]
test_svmVI <- test[features_svmVI]

# RPart

model_rpart_svmVI <- train(class_var ~ ., data = train_svmVI, 
                           method = "rpart", trControl = train_control,
                           tuneLength = 10)

test_pred_rpart_svmVI <- predict(model_rpart_svmVI, newdata = test_svmVI)
confusionMatrix(test_pred_rpart_svmVI, test_svmVI$class_var, mode = "everything")

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

# SVM Radial

SVM_model_svmVI <- train(class_var ~ ., data = train_svmVI, method = "svmRadial",
                         preProc = c("center", "scale"),
                         trControl = train_control, tuneGrid = svmGrid)
test_pred_svm_svmVI <- predict(SVM_model_svmVI, newdata = test_svmVI)
confusionMatrix(test_pred_svm_svmVI, test_svmVI$class_var, mode = "everything")

# Bagged ADAboost

model_adaboost_svmVI <- train(class_var ~ ., data = train_svmVI, method = "AdaBag",
                              trControl = train_control_SVM)

test_pred_adaboost_svmVI <- predict(model_adaboost_svmVI, newdata = test_svmVI)
confusionMatrix(test_pred_adaboost_svmVI, test_svmVI$class_var, mode = "everything")

