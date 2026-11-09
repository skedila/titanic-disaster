# Titanic survival prediction with logistic regression (R).
#
# Loads train.csv / test.csv, cleans and engineers features, trains a
# logistic regression model (glm), reports accuracy, and writes predictions.
# Every step prints to the terminal so the run can be followed without
# reading the code.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

# In Docker, DATA_DIR=/app/data (set in the Dockerfile).
# Running locally from the repo root, default to src/data.
data_dir <- Sys.getenv("DATA_DIR", unset = "src/data")
train_path <- file.path(data_dir, "train.csv")
test_path <- file.path(data_dir, "test.csv")
gender_path <- file.path(data_dir, "gender_submission.csv")
output_path <- file.path(data_dir, "predictions_r.csv")

common_titles <- c("Mr", "Mrs", "Miss", "Master")

section <- function(title) {
  cat("\n", strrep("=", 60), "\n", title, "\n", strrep("=", 60), "\n", sep = "")
}

for (p in c(train_path, test_path)) {
  if (!file.exists(p)) {
    cat("ERROR:", p, "not found.\n")
    cat("Download the data from Kaggle and place it in src/data/ (see README).\n")
    quit(status = 1)
  }
}

extract_title <- function(names) {
  t <- trimws(sub(".*,\\s*([^\\.]+)\\..*", "\\1", names))
  t <- recode(t, Mlle = "Miss", Ms = "Miss", Mme = "Mrs")
  ifelse(t %in% common_titles, t, "Rare")
}

clean_and_engineer <- function(df, fill, label) {
  cat(sprintf("\n[%s] Missing values before cleaning:\n", label))
  print(colSums(is.na(df[, c("Age", "Fare", "Embarked", "Cabin")])))

  n_age <- sum(is.na(df$Age))
  df <- df %>%
    left_join(fill$age_by_title, by = "Title") %>%
    mutate(Age = coalesce(Age, TitleAge, fill$age_overall)) %>%
    select(-TitleAge)
  cat(sprintf("[%s] Filled %d missing Age values with median age by Title\n", label, n_age))

  n_fare <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- fill$fare
  cat(sprintf("[%s] Filled %d missing Fare values with training median (%.2f)\n", label, n_fare, fill$fare))

  n_emb <- sum(is.na(df$Embarked))
  df$Embarked[is.na(df$Embarked)] <- fill$embarked
  cat(sprintf("[%s] Filled %d missing Embarked values with mode ('%s')\n", label, n_emb, fill$embarked))

  pct_cabin <- mean(is.na(df$Cabin))
  df <- df %>%
    mutate(
      Sex = factor(Sex, levels = c("female", "male")),
      Pclass = factor(Pclass, levels = c(1, 2, 3)),
      Embarked = factor(Embarked, levels = c("C", "Q", "S")),
      Title = factor(Title, levels = c(common_titles, "Rare")),
      FamilySize = SibSp + Parch + 1,
      IsAlone = as.integer(FamilySize == 1),
      HasCabin = as.integer(!is.na(Cabin))
    ) %>%
    select(-Name, -Ticket, -Cabin)
  cat(sprintf("[%s] Converted Sex, Pclass, Embarked and Title to factors (fixed levels)\n", label))
  cat(sprintf("[%s] Added FamilySize (SibSp + Parch + 1) and IsAlone\n", label))
  cat(sprintf("[%s] Added HasCabin flag (Cabin is %.0f%% missing, too sparse to use directly)\n",
              label, 100 * pct_cabin))
  cat(sprintf("[%s] Dropped Name, Ticket and Cabin\n", label))
  df
}

# ---- Load ------------------------------------------------------------------
section("STEP 1: Loading training data")
train <- read_csv(train_path, show_col_types = FALSE)
cat(sprintf("Loaded train.csv: %d rows, %d columns\n", nrow(train), ncol(train)))
cat("\nFirst 5 rows:\n")
print(head(as.data.frame(train), 5))

# ---- Explore ---------------------------------------------------------------
section("STEP 2: Exploring training data")
cat(sprintf("Overall survival rate: %.3f\n", mean(train$Survived)))
cat("\nSurvival rate by Sex:\n")
print(as.data.frame(train %>% group_by(Sex) %>% summarise(rate = round(mean(Survived), 3))))
cat("\nSurvival rate by Pclass:\n")
print(as.data.frame(train %>% group_by(Pclass) %>% summarise(rate = round(mean(Survived), 3))))

# ---- Adjust ----------------------------------------------------------------
section("STEP 3: Cleaning and feature engineering")
train$Title <- extract_title(train$Name)
cat("Extracted Title from Name. Counts:\n")
print(table(train$Title))

# Imputation values are learned from the training set only
fill <- list(
  age_by_title = train %>% group_by(Title) %>% summarise(TitleAge = median(Age, na.rm = TRUE)),
  age_overall = median(train$Age, na.rm = TRUE),
  fare = median(train$Fare, na.rm = TRUE),
  embarked = names(which.max(table(train$Embarked)))
)
train_clean <- clean_and_engineer(train, fill, "train")

# FamilySize replaces SibSp and Parch: it is their exact sum (+1), and glm
# cannot estimate all three at once (perfect collinearity)
model_formula <- Survived ~ Pclass + Sex + Age + Fare +
  FamilySize + IsAlone + HasCabin + Embarked + Title
cat("\nModel formula:\n")
print(model_formula)

# ---- Model + training accuracy --------------------------------------------
section("STEP 4: Training logistic regression")
set.seed(42)
val_idx <- sample(seq_len(nrow(train_clean)), size = round(0.2 * nrow(train_clean)))
holdout_model <- glm(model_formula, data = train_clean[-val_idx, ], family = binomial)
val_pred <- as.integer(predict(holdout_model, train_clean[val_idx, ], type = "response") > 0.5)
val_acc <- mean(val_pred == train_clean$Survived[val_idx])
cat(sprintf("Hold-out validation accuracy (80/20 split): %.4f\n", val_acc))

model <- glm(model_formula, data = train_clean, family = binomial)
train_pred <- as.integer(predict(model, train_clean, type = "response") > 0.5)
train_acc <- mean(train_pred == train_clean$Survived)
cat(sprintf("Refit on full training set. Training accuracy: %.4f\n", train_acc))

cat("\nModel coefficients (positive = higher survival odds):\n")
print(round(sort(coef(model)), 3))

# ---- Test predictions -----------------------------------------------------
section("STEP 5: Predicting on the test set")
test <- read_csv(test_path, show_col_types = FALSE)
cat(sprintf("Loaded test.csv: %d rows, %d columns\n", nrow(test), ncol(test)))
test$Title <- extract_title(test$Name)
test_clean <- clean_and_engineer(test, fill, "test")
test_pred <- as.integer(predict(model, test_clean, type = "response") > 0.5)
cat(sprintf("\nPredicted survivors: %d of %d (%.3f)\n", sum(test_pred), length(test_pred), mean(test_pred)))

write_csv(data.frame(PassengerId = test$PassengerId, Survived = test_pred), output_path)
cat("Saved predictions to predictions_r.csv in the data folder\n")

# ---- Test accuracy --------------------------------------------------------
section("STEP 6: Test set accuracy")
cat("Note: Kaggle's test.csv does not include the Survived column,\n")
cat("so true test accuracy can only be obtained by submitting to Kaggle.\n")
if (file.exists(gender_path)) {
  ref <- read_csv(gender_path, show_col_types = FALSE)
  ref <- ref$Survived[match(test$PassengerId, ref$PassengerId)]
  cat(sprintf("Agreement with gender_submission.csv (Kaggle's 'women survive' baseline): %.4f\n",
              mean(ref == test_pred)))
} else {
  cat("gender_submission.csv not found; skipping baseline comparison.\n")
}
cat(sprintf("Best offline estimate of test accuracy (hold-out validation): %.4f\n", val_acc))

section("DONE")
