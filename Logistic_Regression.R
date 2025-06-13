# ===================================================================
#                  AI and Biotechnology/Bioinformatics        
# ===================================================================

# ===================================================================
#                Logistic Regression: Pima Indians Dataset
# ===================================================================

# Load required libraries
library(mlbench)   # For the PimaIndiansDiabetes dataset
library(caret)     # For data splitting and model evaluation (confusion matrix)
library(pROC)      # For ROC and AUC computation
library(ggplot2)   # For plotting and visualization

# Load dataset
data("PimaIndiansDiabetes")
data <- PimaIndiansDiabetes

# Explore the dataset
View(data)        # Open dataset in spreadsheet viewer
dim(data)         # Get dimensions (rows, columns)
str(data)         # View structure: variable types and sample values
is.na(data)       # Check for missing values (returns logical vector)

# -------------------------
# Data Splitting (Train/Test)
# -------------------------

# Split data: 70% training, 30% testing
# Use createDataPartition to preserve class proportions
set.seed(123)  # for reproducibility
index <- createDataPartition(data$diabetes,
                             p = 0.7, 
                             list = FALSE)
train <- data[index, ]
test <- data[-index, ]

# ----------------------------------------------------
# Fit Logistic Regression Model (With All Predictors)
# ----------------------------------------------------

# Fit logistic regression using glm() with binomial family (binary classification)
model <- glm(diabetes ~ ., 
             data = train, 
             family = "binomial")

# View model summary (coefficients, significance, etc.)
summary(model)

# -----------------
# Model Evaluation 
# -----------------

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

# ROC Curve & AUC
roc <- roc(test$diabetes, predicted_prob)
auc <- auc(roc)
print(auc)
plot(roc)

# ----------------------------------
# Logistic Regression (Glucose Only)
# ----------------------------------

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

# -------------------------
# Visualization: Probability Curve & Scatter Plot
# -------------------------

# Convert diabetes status to numeric (0 = neg, 1 = pos)
test$diabetes_status <- ifelse(test$diabetes == "pos", 1, 0)

# Create new glucose data to plot a smooth probability curve
new_data <- data.frame(
  glucose = seq(min(test$glucose), 
                max(test$glucose), 
                length.out = 300)
)
new_data$prob <- predict(model_glu, new_data, type = "response")

# Plot: Probability curve + jittered scatter plot
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


# follow for more:
# github: https://github.com/AI-Biotechnology-Bioinformatics
# linkedin: https://www.linkedin.com/company/ai-and-biotechnology-bioinformatics/
