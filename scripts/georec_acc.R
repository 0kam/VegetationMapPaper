library(tidyverse)
library(jsonlite)

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

w_dist <- projection_error(
    "data/hand_picked_gcp/with_distortion/handpicked_original.json",
    "data/hand_picked_gcp/with_distortion/handpicked_simulated.json",
    "data/hand_picked_gcp/with_distortion/georectified.csv"
)

h1 <- w_dist %>%
    ggplot(aes(x = error)) +
    geom_histogram()

ggsave("data/hand_picked_gcp/with_distortion/hist_w_dist.png", h1)

w_dist %>%
    summarise(mean(error))

p <- read_json("data/hand_picked_gcp/with_distortion/params_optim.json")

gcp <- read_csv("data/gcp.csv") %>%
    mutate(distance = sqrt((p$x - x)^2 + (p$y - y) + (p$z - z))) %>%
    mutate(type = "GCP")

gcp %>% 
    ggplot(aes(x = distance)) +
    geom_boxplot()

w_dist <- w_dist %>%
    mutate(distance = sqrt((p$x - x_org)^2 + (p$y - y_org) + (p$z - z_org))) 
    
p1 <- ggplot() +
    geom_point(data = w_dist, mapping = aes(x = distance, y = error)) +
    geom_violin(data = gcp, mapping = aes(x = distance, y = 0))

ggsave("data/hand_picked_gcp/with_distortion/scatter.png", p1)
#------------------------------------------------------------------------
wo_dist <- projection_error(
    "data/hand_picked_gcp/wo_distortion/handpicked_original.json",
    "data/hand_picked_gcp/wo_distortion/handpicked_simulated.json",
    "data/hand_picked_gcp/wo_distortion/georectified.csv"
)


h2 <- wo_dist %>%
    ggplot(aes(x = error)) +
    geom_histogram()

ggsave("data/hand_picked_gcp/wo_distortion/hist_wo_dist.png", h2)

wo_dist %>%
    summarise(mean(error))

p <- read_json("data/hand_picked_gcp/with_distortion/params_optim.json")

p2 <- wo_dist %>%
    mutate(distance = sqrt((p$x - x_org)^2 + (p$y - y_org) + (p$z - z_org))) %>% 
    ggplot(aes(x = distance, y = error)) +
    geom_point()

ggsave("data/hand_picked_gcp/wo_distortion/scatter.png", p2)
