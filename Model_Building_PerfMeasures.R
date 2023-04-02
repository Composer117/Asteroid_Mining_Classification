library(caret)
library(rsample)
library(MASS)

asteroid_df <- read.csv("Data/asteroid_dataset.csv")
asteroid_df$class_var <- factor(asteroid_df$class_var)
features <- colnames(asteroid_df)[-36]

# The classification algorithms that we are running for this 
# use case are as follows:
# 1. Decision Tree - RPart
# 2. Decision Tree - J48
# 3. SVM Radial
# 4. Random Forest
# 5. NNet

# First, we will run all algorithms on all variables

set.seed(31)
split <- initial_split(asteroid_df, prop = 0.66, strata = class_var)
train <- training(split)
test <- testing(split)

# Starting with rpart

set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, 
                              summaryFunction = defaultSummary)

model_rpart <- train(class_var ~ ., data = train, 
                     method = "rpart", trControl = train_control,
                     tuneLength = 10)

test_pred_rpart <- predict(model_rpart, newdata = test)
confusionMatrix(test_pred_rpart, test$class_var)

# J48

set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, 
                              summaryFunction = defaultSummary)

model_J48 <- train(class_var ~ ., data = train, method = "J48", 
               trControl = train_control,
               tuneLength = 4)

test_pred_J48 <- predict(model_J48, newdata = test)
confusionMatrix(test_pred_J48, test$class_var)

# SVM - Radial

train_control_SVM <- trainControl(method = "repeatedcv", 
                                  number = 10,
                                  summaryFunction = defaultSummary)
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.05), 
                        C = seq(1.0, 2.0, by = 0.1))

modelSVM <- train(class_var ~ ., data = train, method = "svmRadial",
                  preProc = c("center", "scale"),
                  trControl = train_control_SVM, tuneGrid = svmGrid)

test_pred_SVM <- predict(modelSVM, newdata = test)
confusionMatrix(test_pred_SVM, test$class_var)

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
confusionMatrix(pred_RF, test$class_var)

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
confusionMatrix(test_pred_nnet, test$class_var)

# GBM

#train_control_GBM <- trainControl(method = "CV", number = 10,
#             classProbs = TRUE, 
#             summaryFunction = defaultSummary,
#             savePredictions = TRUE)
###           shrinkage = c(.01, .1),
#             n.minobsinnode = 3)
#
#gbmFit <- train(x = train[features], 
#                y = train$class_var,
#                method = "gbm",
#                #tuneGrid = gbmGrid,
#                metric = "ROC",
#                verbose = FALSE,
#                trControl = train_control_GBM)


#test_pred_gbm <- predict(gbmFit, newdata = test)
#confusionMatrix(test_pred_gbm, test$class_var)

