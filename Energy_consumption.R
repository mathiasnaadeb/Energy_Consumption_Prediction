## Prediction of Power Usage using five (5) different machine learning models

# Libraries
library(tidyverse)
library(tidymodels)
library(caret)
library(vip)
library(kknn)
library(kernlab)
library(earth)
library(magrittr)
library(lubridate)
library(MASS)


steel_data <- read_csv(here::here("C:/Users/mathi/Documents/Steel_industry_data.csv"))


# Changing the date format
head(steel_data$date)
class(steel_data$date)
class(steel_data)

# Changing the date to tge right class/format
steel_data$date <-ymd_hms(steel_data$date)


# Taking a glimpse of the data
steel_data %>%
  dplyr::glimpse()


# Checking if there are missing values
steel_data %>%
  summarize(across(.cols = everything(),
                   ~sum(is.na(.x))))

# Plotting a time series graph of the power usage against the date
steel_data %>%
  ggplot(aes(x = date, y= Usage_kWh)) +
  geom_point()+
  geom_line()


## Feature selection (Used 10,000 data points because of computational difficulties)
steel_feature <-caret::rfe(x = steel_data[c(1:10000),], y = steel_data$Usage_kWh[c(1:10000)], rfeControl = rfeControl(rfFuncs))
summary(steel_feature)

### Predicting the Power Usage based on the predictors (features) selected from the feature selection 

steel_data <- steel_data[,-c(1,4,7,8,9,10,11)]

#### Split data into train and Test

set.seed(4321) # for reproducibility

# insurance_split <- initial_split(insurance, strat = "region")
steel_split <- initial_split(steel_data)

steel_train <- training(steel_split) # Training data set
steel_test <- testing(steel_split)   # Testing data set

## Cross Validation Split of Training Data
set.seed(4321)
cv_folds <- vfold_cv(
  data = steel_train, 
  v = 5
) 

cv_folds


## Recipe

steel_rec <- recipe(
  Usage_kWh ~ .,
  data = steel_train
) 

steel_rec %>%
  prep() %>%
  bake(new_data = NULL)



## Model Specifications - Using 5 different models
# 1) Random Forest
# 2) Support Vector Machines
# 3) KNN Regression
# 4) Mars
# 5) Linear Regression



# Random Forest
rf_spec <- rand_forest() %>%
  set_mode("regression") %>%
  set_engine("randomForest", importance = TRUE)


# Support Vector Machines 
svm_spec <- svm_rbf() %>%
  set_mode("regression") %>%
  set_engine("kernlab")


# K nearest Neighbors
knn_spec <- nearest_neighbor(neighbors = 4) %>%
  set_mode("regression")


# Multivariate adaptive regression splines (Mars)
mars_spec <- mars() %>%
  set_mode("regression")%>%
  set_engine("earth")


# Linear Regression
lm_spec <- linear_reg() %>% 
  set_engine("lm") %>%
  set_mode("regression")


## Workflow Set

# Here we Combine the pre-processing recipe and the 5 models together

wf_set <-workflow_set(
  preproc = list(steel_rec),
  models = list(lm_spec, rf_spec, knn_spec, svm_spec, mars_spec)
)

wf_set

## Fit the models to the workflow

doParallel::registerDoParallel()

steel_fit <- workflow_map(
  wf_set,"fit_resamples",
  resamples = cv_folds,
  seed = 4321 ## replicability
)

steel_fit


## Evaluate model fits

autoplot(steel_fit)

collect_metrics(steel_fit)

results<-rank_results(steel_fit, rank_metric = "rmse", select_best = TRUE)



## Extracting the best model workflow for predicting the Power Usage
wf_final <- extract_workflow(steel_fit, id = results$wflow_id[1]) %>%
  fit(steel_train)

wf_final

## Prediction on test dataset

steel_preds <- predict(
  wf_final %>% extract_fit_parsnip(), 
  new_data = steel_rec %>% prep() %>% bake(new_data =steel_test)
)

steel_preds %>% head()

test_final <- cbind(steel_test, steel_preds)
test_final %>% head()


res_plot <- test_final %>%
  ggplot(aes(x = .pred, y = Usage_kWh))+
  geom_smooth(method = "lm",
              color = "red",
              size = 2) +
  geom_abline(intercept = 0,
              slope = 1,
              size = 2,
              color = "green",
              linetype = "dashed") +
  ggtitle("Usage_kWh")

res_plot


## all in one
wf_final_full <- extract_workflow(steel_fit, results$wflow_id[1]) %>%
  last_fit(steel_split)

wf_final_full %>% collect_predictions()


