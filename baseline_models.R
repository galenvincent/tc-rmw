library(tidyverse)
library(lubridate)
library(ggcorrplot)

data <- read.csv('data/regression_data_pca_lagged_quad.csv') %>%
  mutate(atcf = as.logical(atcf)) %>%
  mutate(date = ymd_hms(date))


# Subset to ATCF
data_atcf <- data %>% filter(atcf == T)

# Split data into training and testing by year
train_data <- data_atcf %>% filter(year(date) < 2017)
test_data <- data_atcf %>% filter(year(date) >= 2017)

# Linear baseline using no-lag PCs:
mod_1 <- lm(rmw ~ lat + lon + wind + pressure_min + distance + age + 
              rad_pc_1 + rad_pc_2 + rad_pc_3 + size_pc_1 + size_pc_2 + size_pc_3,
            data = train_data)
summary(mod_1)

# Get RMSE on test data
modelr::rmse(mod_1, test_data)

# Does adding lagged values help? Perform an ANOVA to see:
mod_full <- lm(rmw ~ lat + lon + wind + pressure_min + distance + age + 
                rad_pc_1 + rad_pc_2 + rad_pc_3 + size_pc_1 + size_pc_2 + size_pc_3 +
                rad_pc_1_6p + rad_pc_2_6p + rad_pc_3_6p,
            data = data_atcf %>% filter(!is.na(rad_pc_1_6p)))

mod_red <- lm(rmw ~ lat + lon + wind + pressure_min + distance + age + 
                 rad_pc_1 + rad_pc_2 + rad_pc_3 + size_pc_1 + size_pc_2 + size_pc_3,
               data = data_atcf %>% filter(!is.na(rad_pc_1_6p)))
anova(mod_full, mod_red)

# Model with quadrants:
mod_2 <- lm(rmw ~ lat + lon + wind + pressure_min + distance + age + 
              size_pc_1 + size_pc_2 + size_pc_3 +
              rad_pc_1_NE + rad_pc_1_NW + rad_pc_1_SW + rad_pc_1_SE +
              rad_pc_2_NE + rad_pc_2_NW + rad_pc_2_SW + rad_pc_2_SE +
              rad_pc_3_NE + rad_pc_3_NW + rad_pc_3_SW + rad_pc_3_SE,
            data = train_data)

modelr::rmse(mod_2, test_data)

# Correlation between lagged values
data %>% 
  select(rad_pc_1_6m, rad_pc_1_4m, rad_pc_1_2m, rad_pc_1_1m, rad_pc_1_1p, rad_pc_1_2p, rad_pc_1_4p, rad_pc_1_6p) %>%
  drop_na() %>%
  cor() %>%
  ggcorrplot()

data %>% 
  select(rad_pc_2_6m, rad_pc_2_4m, rad_pc_2_2m, rad_pc_2_1m, rad_pc_2_1p, rad_pc_2_2p, rad_pc_2_4p, rad_pc_2_6p) %>%
  drop_na() %>%
  cor() %>%
  ggcorrplot()
  
data %>% 
  select(rad_pc_3_6m, rad_pc_3_4m, rad_pc_3_2m, rad_pc_3_1m, rad_pc_3_1p, rad_pc_3_2p, rad_pc_3_4p, rad_pc_3_6p) %>%
  drop_na() %>%
  cor() %>%
  ggcorrplot()

data %>% 
  select(size_pc_1_6m, size_pc_1_4m, size_pc_1_2m, size_pc_1_1m, size_pc_1_1p, size_pc_1_2p, size_pc_1_4p, size_pc_1_6p) %>%
  drop_na() %>%
  cor() %>%
  ggcorrplot()

data %>% 
  select(size_pc_2_6m, size_pc_2_4m, size_pc_2_2m, size_pc_2_1m, size_pc_2_1p, size_pc_2_2p, size_pc_2_4p, size_pc_2_6p) %>%
  drop_na() %>%
  cor() %>%
  ggcorrplot()

data %>% 
  select(size_pc_3_6m, size_pc_3_4m, size_pc_3_2m, size_pc_3_1m, size_pc_3_1p, size_pc_3_2p, size_pc_3_4p, size_pc_3_6p) %>%
  drop_na() %>%
  cor() %>%
  ggcorrplot()


# Correlation between quadrants
data %>%
  select(rad_pc_1_NE, rad_pc_1_NW, rad_pc_1_SW, rad_pc_1_SE) %>%
  drop_na() %>%
  cor() %>%
  ggcorrplot()

data %>%
  select(rad_pc_2_NE, rad_pc_2_NW, rad_pc_2_SW, rad_pc_2_SE) %>%
  drop_na() %>%
  cor() %>%
  ggcorrplot()

data %>%
  select(rad_pc_3_NE, rad_pc_3_NW, rad_pc_3_SW, rad_pc_3_SE) %>%
  drop_na() %>%
  cor() %>%
  ggcorrplot()
