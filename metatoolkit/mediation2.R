# Step 1: Simulate Data (Same as before)
set.seed(123)

n <- 100
X <- sample(0:1, n, replace = TRUE)
M <- 0.5 * X + rnorm(n, mean = 0, sd = 1)
Y <- 2 * X + 0.3 * M + rnorm(n, mean = 0, sd = 2)
data <- data.frame(X = X, M = M, Y = Y)

# Step 2: Fit the Models (Same as before)
mediator_model <- lm(M ~ X, data = data)
outcome_model <- lm(Y ~ X + M, data = data)

# Step 3: Perform Mediation Analysis (Same as before)
library(mediation)
library(igraph)
mediation_result <- mediate(mediator_model, outcome_model, treat = "X", mediator = "M", boot = TRUE, sims = 1000)

# Step 4: Extract Coefficients for Network Plot
coef_X_M <- coef(mediator_model)["X"]
coef_M_Y <- coef(outcome_model)["M"]
coef_X_Y <- coef(outcome_model)["X"]

# Step 5: Create a Network Data Frame
# Nodes: X, M, Y
# Edges: Relationships between them with coefficients as weights

nodes <- data.frame(name = c("X", "M", "Y"))

edges <- data.frame(
  from = c("X", "M", "X"), 
  to = c("M", "Y", "Y"),
  weight = c(coef_X_M, coef_M_Y, coef_X_Y)
)

# Step 6: Create the Graph
graph <- graph_from_data_frame(edges, vertices = nodes, directed = TRUE)

# Step 7: Visualize the Network with ggraph
library(ggraph)
ggraph(graph, layout = "circle") +
  geom_edge_link(aes(edge_alpha = weight, edge_width = weight), color = "darkblue") +
  geom_node_point(color = "skyblue", size = 6) +
  geom_node_text(aes(label = name), color = "black", size = 6, fontface = "bold") +
  theme_void() +
  labs(title = "Mediation Analysis Network: X → M → Y") +
  theme(legend.position = "none")

