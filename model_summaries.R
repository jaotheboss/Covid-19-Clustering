# setting up the work directory
setwd("~/Documents/Active Projects/Coronavirus Prediction (Kaggle)")

# importing relevant libraries
library(tidyverse)

# importing data
data = read_csv('data.csv')
data %>% select(-c('Country', 'Cluster'))

# splitting into cases and fatalities
case_data = data %>% select(-'Fatalities')
fata_data = data %>% select(-'Cases')

case_model = lm(Cases ~ ., data = case_data, )
summary(case_model)

fata_model = lm(Fatalities ~ ., data = fata_data)
summary(fata_model)
anova(fata_model)
