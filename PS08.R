library(tidyverse)
library(caret)
library(MLmetrics)
# Package for easy timing in R
library(tictoc)



# Demo of timer function --------------------------------------------------
# Run the next 5 lines at once
tic()
Sys.sleep(3)
timer_info <- toc()
runtime <- timer_info$toc - timer_info$tic
runtime



# Get data ----------------------------------------------------------------
# Accelerometer Biometric Competition Kaggle competition data
# https://www.kaggle.com/c/accelerometer-biometric-competition/data
train <- read_csv("~/Downloads/train.csv")

# YOOGE!
dim(train)



# knn modeling ------------------------------------------------------------
model_formula <- as.formula(Device ~ X + Y + Z)

# Values to use:
n_values <- c(10, 20, 30)
k_values <- c(2, 3, 4)

runtime_dataframe <- expand.grid(n_values, k_values) %>%
  as_tibble() %>%
  rename(n=Var1, k=Var2) %>%
  mutate(runtime = n*k)
runtime_dataframe


# Time knn here -----------------------------------------------------------
runtime_df <- matrix(0,2000,3)
for(i in 1:500){
  n = 1000*i
  loopcounter = i*4-4
  sample_train <- train %>% slice(1:n)
  k_values = c(2,10,100,1000,10000)
  for(j in 1:4){
    tic()
    model_knn <- caret::knn3(model_formula, data= sample_train, k = k_values[j])
    tim <- toc()
    runtime_df[loopcounter + j,1] <- n
    runtime_df[loopcounter + j,2] <- k_values[j]
    runtime_df[loopcounter + j,3] <- tim$toc - tim$tic
  }
}

runtime_df <- as.data.frame(runtime_df)
runtime_df <- runtime_df %>% rename(n = V1, k = V2, time = V3)
write_csv(runtime_df, "results.csv")

# Plot your results ---------------------------------------------------------
# Think of creative ways to improve this barebones plot. Note: you don't have to
# necessarily use geom_point
runtime_plot <- ggplot(runtime_df, aes(x=n, y=time)) +
  geom_point(aes(color = k)) + ggtitle("Comparing runtime to k and n for knn")
runtime_plot

plot2 <- ggplot(runtime_df, aes(x=k, y=time)) + geom_point()
plot2

plot3 <- ggplot(runtime_df, aes(x=n, y=time)) +geom_point()
plot3

ggsave(filename="jennifer_halbleib.png", plot = runtime_plot, width=16, height = 9)

# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:
# -n: number of points in training set
# O(n)
# -k: number of neighbors to consider
# O(1)
# -d: number of predictors used? In this case d is fixed at 3
# O(d*n) so potentially exponential if d = n


