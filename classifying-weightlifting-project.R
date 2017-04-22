options(scipen = 999)

# Packages
library(data.table)

if (!require("caret")) {
  install.packages("caret")
  library("caret")
}

if (!require("randomForest")) {
  install.packages("randomForest")
  library("randomForest")
}

if (!require("rpart")) {
  install.packages("rpart")
  library("rpart")
}

if (!require("rpart.plot")) {
  install.packages("rpart.plot")
  library("rpart.plot")
}

if (!require("ggplot2")) {
  install.packages("ggplot2")
  library("ggplot2")
}

if (!require("e1071")) {
  install.packages('e1071', dependencies=TRUE)
  library("e1071")
}
# Reproducability
set.seed(9999)


# Data
training_file_name   <- './data/pml-training.csv'
testing_file_name <- './data/pml-testing.csv'

# Directories
setwd("C:\\DataCamp\\prediction-assignment-writeup")
if (!file.exists("data")) (dir.create("data"))
if (!file.exists("submission")) dir.create("submission")

if(!file.exists(training_file_name)) 
  download.file('http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', training_file_name, quiet = FALSE)

if(!file.exists(testing_file_name)) 
  download.file('http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', testing_file_name, quiet = FALSE)

# Load datasets
na.strings <- c("NA","#DIV/0!", "")
pml_training <- fread(training_file_name, na.strings = na.strings, sep = ",")
pml_testing <-fread(testing_file_name , na.strings = na.strings)

pml_training <- pml_training[,-c(1:7)]
pml_testing  <- pml_testing[,-c(1:7)]  

# Data cleaning
pml_training <- pml_training[, colSums(is.na(pml_training)) == 0, with = FALSE]
pml_testing <- pml_testing[, colSums(is.na(pml_testing)) == 0, with = FALSE]


# Spliting training and testing sets
indexes <- createDataPartition(y = pml_training$classe, p=0.6, list = FALSE)
training_set <- pml_training[indexes, ] 
testing_set <- pml_training[-indexes, ]


# Exploratory analysis
ggplot(training_set,aes(x = classe, fill = classe)) + geom_bar() + scale_color_brewer(palette="Spectral")

## Prediction models


### Decision Tree

# Train Classification Tree Model
tree_model <- rpart(classe ~ ., data = training_set, method="class")

# Plot Classification Tree
rpart.plot(tree_model, main="Classification Tree", extra=102, under=TRUE, faclen=0)

# Prediction Classification Tree model
tree_model_predicted <- predict(tree_model, testing_set, type = "class")

# Show Classification Tree confusion matrix
confusion_matrix_tree_model <- confusionMatrix(tree_model_predicted, testing_set$classe)
confusion_matrix_tree_model
tree_model_error <- 1 - confusion_matrix_tree_model$overall['Accuracy']
sprintf("Classification Tree Model Error: %3.2f", tree_model_error * 100)
### Random forest

# Fit model
train_control <- trainControl(method="cv", number=5, verboseIter = FALSE , preProcOptions="pca")
random_forest_model <- train(classe ~ ., data = training_set, method="rf", trControl = train_control, ntree = 50)

# Perform prediction
random_forest_predicted <- predict(random_forest_model, testing_set, type = "raw")

# Show Random forest confusion matrix
confusion_matrix_random_forest <- confusionMatrix(random_forest_predicted, testing_set$classe)
random_forest_error <- 1 - confusion_matrix_random_forest$overall['Accuracy']
sprintf("Classification Tree Model Error: %3.2f", random_forest_error * 100)

### Test Case
final_predict <- predict(random_forest_model, pml_testing)

write.csv(as.data.table(final_predict), "./submission/submission.csv")

