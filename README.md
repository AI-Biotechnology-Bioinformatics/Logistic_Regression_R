# Logistic Regression
Logistic Regression is a supervised machine learning algorithm commonly used for binary classification tasks. It is simple, interpretable, and widely used in health and biological data analysis. Logistic Regression predicts the probability of outcomes like 0/1, Yes/No, or Healthy/Diseased.
It uses the sigmoid(logit) function to map predicted values between 0 and 1, henece the named logistic regression.
Logistic Regression identifies a linear boundary that separates two classes. The sigmoid function then maps any real-valued input to a value between 0 and 1, which can be interpreted as a probability. If the predicted probability is greater than 0.5, the observation is classified as 1 (e.g., diseased); otherwise, it is classified as 0 (e.g., healthy).

To perform logistic regression in R, use the glm() function with family = "binomial": for binary classification task.
> **Watch the full step-by-step tutorial** → [Click Here]() ,  
> **Download the → [R Script](https://github.com/AI-Biotechnology-Bioinformatics/Logistic_Regression_R/blob/main/Logistic_Regression.R)** ,     
> **Follow the below steps to perform Logistic Regression in R.**

In this analysis, we’ll predict diabetes status (pos or neg) using clinical features from the PimaIndiansDiabetes dataset.
## Loading Required Libraries
```r
library(mlbench)   # For the PimaIndiansDiabetes dataset
library(caret)     # For data splitting and model evaluation (confusion matrix)
library(pROC)      # For ROC and AUC computation
library(ggplot2)   # For plotting and visualization
```
## Load dataset
```r
data("PimaIndiansDiabetes")
data <- PimaIndiansDiabetes
```

## Explore the dataset
```r
View(data)        # Open dataset in spreadsheet viewer
dim(data)         # Get dimensions (rows, columns)
str(data)         # View structure: variable types and sample values
is.na(data)       # Check for missing values (returns logical vector)
```
## Data Splitting (Train/Test)
```r
# Split data: 70% training, 30% testing
# Use createDataPartition to preserve class proportions
set.seed(123)  # for reproducibility
index <- createDataPartition(data$diabetes,
                             p = 0.7, 
                             list = FALSE)
train <- data[index, ]
test <- data[-index, ]
```
## Fit Logistic Regression Model (With All Predictors)
```r
# Fit logistic regression using glm() with binomial family (binary classification)
model <- glm(diabetes ~ ., 
             data = train, 
             family = "binomial")

# View model summary (coefficients, significance, etc.)
summary(model)
```
## Model Evaluation 
```r
# Predict probabilities for test set
predicted_prob <- predict(model, 
                          newdata = test, 
                          type = "response")

# Convert probabilities to class labels using 0.5 cutoff
# If the probability is greater than 0.5, if classify as "pos"

predicted_class <- ifelse(predicted_prob > 0.5, "pos", "neg")

# Generate confusion matrix
conf <- confusionMatrix(factor(predicted_class), test$diabetes)
print(conf)

# Misclassification Rate
1 - conf$overall["Accuracy"]
```
## ROC Curve & AUC
```r
roc <- roc(test$diabetes, predicted_prob)
auc <- auc(roc)
print(auc)
plot(roc)
```
# Logistic Regression (Glucose Only)
```r
# Fit logistic regression with glucose as a single predictor
# similarly you can add more than one predictors 
# glm(diabetes ~ glucose + mass + pregnancy .....)

model_glu <- glm(diabetes ~ glucose, 
                 data = train, 
                 family = "binomial")
summary(model_glu)

# Predict probabilities and classes for test set
prob <- predict(model_glu, 
                test, 
                type = "response")
class <- ifelse(prob > 0.5, "pos", "neg")

# Confusion matrix
conf_glu <- confusionMatrix(factor(class), test$diabetes)
print(conf_glu)

# ROC & AUC
roc_glu <- roc(test$diabetes, prob)
auc_glu <- auc(roc_glu)
plot(roc_glu)
```
## Visualization: Probability Curve & Scatter Plot
```r
# Convert diabetes status to numeric (0 = neg, 1 = pos)
test$diabetes_status <- ifelse(test$diabetes == "pos", 1, 0)

# Create new glucose data to plot a smooth probability curve
new_data <- data.frame(
  glucose = seq(min(test$glucose), 
                max(test$glucose), 
                length.out = 300)
)
new_data$prob <- predict(model_glu, new_data, type = "response")
```
## Plot: Probability curve + scatter plot
```r
ggplot(data = test, aes(x = glucose, 
                        y = diabetes_status)) +
  geom_line(data = new_data, aes(x = glucose, 
                                 y = prob), color = "black", size = 1) +  # Probability curve
  geom_jitter(width = 0, height = 0.05, size = 3, alpha = 0.4, color = "blue") +        # Jittered scatter points
  theme_bw() +
  labs(
    title = "Logistic Regression: Glucose level vs Diabetes Status",
    x = "Glucose Level",
    y = "Diabetes Status (0 = Neg, 1 = Pos)"
  )
```
