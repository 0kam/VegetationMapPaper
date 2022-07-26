library(tidyverse)
library(jsonlite)

setwd("~/VegetationMapPaper/")

as_df <- function(pt) {
  u <- as.integer(pt$points[[1]][[1]])
  v <- as.integer(pt$points[[1]][[2]])
  label <- pt$label %>% as.integer()
  return(tibble(u = u, v = v, label=label))
}

projection_error <- function(org_json, sim_json, georec_csv) {
  org <- read_json(org_json)
  sim <- read_json(sim_json)
  org <- org$shapes %>%
    map_dfr(as_df) %>% 
    arrange(label)
  sim <- sim$shapes %>%
    map_dfr(as_df) %>%
    arrange(label)
  all(sim$label == org$label)
  gr <- read_csv(georec_csv) %>%
    select(-c(B, G, R))
  org2 <- org %>%
    inner_join(gr, by = c("u", "v")) %>%
    select(-c(u, v)) %>%
    rename(
      x_org = x,
      y_org = y,
      z_org = z
    )
  sim2 <- sim %>%
    inner_join(gr, by = c("u", "v")) %>%
    select(-c(u, v)) %>%
    rename(
      x_sim = x,
      y_sim = y,
      z_sim = z
    )
  df <- inner_join(org2, sim2, by = "label") %>%
    mutate(error = sqrt((x_org - x_sim)^2 + (y_org - y_sim)^2 + (z_org - z_sim)^2)) %>%
    arrange(desc(error))
  return(df)
}

p <- read_json("data/hand_picked_gcp/with_distortion/params_optim.json")

w_dist <- projection_error(
  "data/hand_picked_gcp/with_distortion/handpicked_original.json",
  "data/hand_picked_gcp/with_distortion/handpicked_simulated.json",
  "data/hand_picked_gcp/with_distortion/georectified.csv"
) %>%
  mutate(distance = sqrt((p$x - x_org)^2 + (p$y - y_org) + (p$z - z_org))) %>%
  mutate(method = "Proposed Method")

wo_dist <- projection_error(
  "data/hand_picked_gcp/wo_distortion/handpicked_original.json",
  "data/hand_picked_gcp/wo_distortion/handpicked_simulated.json",
  "data/hand_picked_gcp/wo_distortion/georectified.csv"
) %>% 
  mutate(distance = sqrt((p$x - x_org)^2 + (p$y - y_org) + (p$z - z_org))) %>%
  mutate(method = "Without Distortion")

silh <- projection_error(
  "data/hand_picked_gcp/silhouette/handpicked_original.json",
  "data/hand_picked_gcp/silhouette/handpicked_simulated.json",
  "data/hand_picked_gcp/silhouette/georectified.csv"
) %>%
  mutate(distance = sqrt((p$x - x_org)^2 + (p$y - y_org) + (p$z - z_org))) %>%
  mutate(method = "Silhouette")

df <- bind_rows(
  w_dist,
  wo_dist,
  silh
)

df %>%
  write_csv("data/georec_acc.csv")

df <- read_csv("data/georec_acc.csv")

p1 <- df %>%
  ggplot(aes(x = distance, y = error, color = method)) +
  geom_point() +
  ylab("Projection Error (m)") +
  xlab("Distance from the Camera (m)") +
  labs(color = "Method") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 18),
    text = element_text(size = 20),
    axis.text.x = element_text(angle = 60, hjust = 1),
    plot.background = element_rect(fill = "white"),
    panel.spacing = grid::unit(2, "lines")
  )


p2 <- df %>%
  filter(method == "Proposed Method") %>%
  ggplot(aes(x = distance, y = error)) +
  geom_point() +
  ylab("Projection Error (m)") +
  xlab("Distance from the Camera (m)") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 18),
    text = element_text(size = 20),
    axis.text.x = element_text(angle = 60, hjust = 1),
    plot.background = element_rect(fill = "white"),
    panel.spacing = grid::unit(2, "lines")
  )

ggsave(file = "results/georec_acc_all.png", plot = p1, width = 12, height = 8)
ggsave(file = "results/georec_acc.png", plot = p2, width = 12, height = 8)


df %>%
  group_by(method) %>%
  summarise(error = mean(error))
  