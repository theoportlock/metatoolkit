#!/usr/bin/env R
# Step 1: Simulate Data

set.seed(123)

# Simulate data for 100 individuals
n <- 100

# Independent variable X (Health status: 0 = unhealthy, 1 = healthy)
X <- sample(0:1, n, replace = TRUE)

# Mediator M (Gut bacteria abundance, simulated as a normal variable influenced by X)
M <- 0.5 * X + rnorm(n, mean = 0, sd = 1)

# Outcome Y (Cognitive function score, influenced by X and M)
Y <- 2 * X + 0.3 * M + rnorm(n, mean = 0, sd = 2)

# Create a data frame
data <- data.frame(X = X, M = M, Y = Y)

# Step 2: Fit the Models

# Model 1 (Mediator Model): How X influences M
mediator_model <- lm(M ~ X, data = data)

# Model 2 (Outcome Model): How X and M influence Y
outcome_model <- lm(Y ~ X + M, data = data)

# Step 3: Perform Mediation Analysis

# Load the mediation package
if (!require(mediation)) install.packages("mediation")
library(mediation)

# Perform mediation analysis
mediation_result <- mediate(mediator_model, outcome_model, 
                            treat = "X", mediator = "M", 
                            boot = TRUE, sims = 1000)

# View results
print(summary(mediation_result))

# Step 4: Visualize the Mediation Path

# Load ggplot2 for visualization
if (!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)

# Plot the relationship between X and M (mediator)
ggplot(data, aes(x = X, y = M)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(title = "Effect of Health Status (X) on Gut Bacteria (M)",
       x = "Health Status (X)",
       y = "Gut Bacteria Abundance (M)") +
  theme_minimal()

# Plot the relationship between M and Y (outcome)
ggplot(data, aes(x = M, y = Y)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Effect of Gut Bacteria (M) on Cognitive Function (Y)",
       x = "Gut Bacteria Abundance (M)",
       y = "Cognitive Function (Y)") +
  theme_minimal()


