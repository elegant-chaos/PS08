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
runtime_df <- matrix(0,20,3)
n_values = c(1000,10000,100000,1000000,10000000)
for(i in 1:length(n_values)){
  loopcounter = i*4-4
  sample_train <- train %>% slice(1:n_values[i])
  k_values = c(2,10,100,1000,10000,100000)
  for(j in 1:length(k_values)){
    tic()
    model_knn <- caret::knn3(model_formula, data= sample_train, k = k_values[j])
    tim <- toc()
    runtime_df[loopcounter + j,1] <- n_values[i]
    runtime_df[loopcounter + j,2] <- k_values[j]
    runtime_df[loopcounter + j,3] <- tim$toc - tim$tic
  }
}

runtime_df <- as.data.frame(runtime_df)
runtime_df <- runtime_df %>% rename(n = V1, k = V2, time = V3)

#Writing results to .csv for making graphs without running program
#write_csv(runtime_df, "results.csv")

# Plot your results ---------------------------------------------------------
# Think of creative ways to improve this barebones plot. Note: you don't have to
# necessarily use geom_point

runtime_plot <- ggplot(runtime_df, aes(x=n, y=time)) +
  geom_point(aes(color = k), size=5) + ggtitle("Comparing runtime to k and n for knn") + geom_smooth(method="lm")
runtime_plot

plot2 <- ggplot(runtime_df, aes(x=k, y=time)) + geom_point()
plot2

plot3 <- ggplot(runtime_df, aes(x=n, y=time)) +geom_point()
plot3 +
  scale_y_log10() +
  scale_x_log10()

ggsave(filename="jennifer_halbleib.png", plot = runtime_plot, width=16, height = 9)

# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:
#Note: I interpretted the following questions as "What is n (or k or d) contributing
# to the overall runtime.
# -n: number of points in training set
#It appears to me, based on the times I've observed, that the process of traversing n
# happens only once in the algorithm, so the contribution of n to runtime is linear.
# O(n)
# -k: number of neighbors to consider
# Based on the data, k does not seem to significant impact runtime, so the contribution
# of k to runtime is constant.
# O(1)
# -d: number of predictors used? In this case d is fixed at 3
# While d is fixed in this case, my case is the algorithm works via some type of 
# matrix traversal. So, I imagine d's contribution to runtime is also linear. This means
# that, overall, the likely worst case scenario is exponential runtime, when d = n.
# O(d*n) so potentially exponential if d = n

