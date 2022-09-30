library(tidyverse)
library(zoo) # moving averages
library(lubridate)
library(hrbrthemes) # themes for graphs
library(geofacet) # maps
library(usmap) # lat and long
library(socviz) # for %nin%
library(ggmap) # mapping


county_adjacency <-  read_table("~/Downloads/county_adjacency.txt")
population_data <- read_csv("~/Downloads/PopulationEstimates.csv")
population_data = population_data %>%
  filter(Attribute %in% c( "Population 2020",  "Population 2021"))
population_data["year"] = 0
for (i in 1:nrow(population_data)){
  if (population_data[i, "Attribute"]  == "Population 2020" ){
    population_data[i, "year"] = 2020
  }
  if (population_data[i, "Attribute"]  == "Population 2021" ){
    population_data[i, "year"] = 2021
  }
}
population_data = population_data %>% rename(fips=FIPStxt)

data2022 <- read_csv(file="https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2022.csv", 
                      )
data2021 <-  read_csv(file="https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2021.csv", 
                     )
data2020 <-  read_csv(file="https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2020.csv", 
                     )



mask_used_per_county = read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/mask-use/mask-use-by-county.csv")
mask_used_per_county = mask_used_per_county %>% rename(fips=COUNTYFP)

data_illinois22 =  data2022 %>% filter(state=='Illinois')
data_illinois21 =  data2021 %>% filter(state=='Illinois')
data_illinois20 =  data2020 %>% filter(state=='Illinois')
data_illinois = rbind(data_illinois22, data_illinois21, data_illinois20)
data_illinois = data_illinois %>%
  mutate(year=year(ymd(date))) %>%
  mutate(year=ifelse(year>2021, 2021, year))  
### join with population on fips
data_illinois = merge(data_illinois, population_data %>% select(fips,Value, year ), 
                      by.x=c('fips', 'year'), by.y=c('fips','year' ),
                      all.x=TRUE,all.y=FALSE)



ggplot(data_illinois %>% filter( fips %in% 17001:17019)) + 
  geom_line(aes(x=date, y=cases, colour=fips)) +
  theme_bw() ### these are the cumulative counts

data_illinois = data_illinois %>%
  group_by(county, date) %>%
  summarize(cases = sum(cases),
            deaths = sum(deaths),
            pop = sum(Value)) %>%
  mutate(new_cases =  cases - lag(cases, k=1, order_by=date),
         new_deaths =  deaths - lag(deaths, k=1, order_by=date))

data_illinois[which(data_illinois["new_cases"]<0), "new_cases"] = 0
data_illinois[which(data_illinois["new_deaths"]<0), "new_deaths"] = 0
  
ggplot(data_illinois %>% filter( date>ymd("2022-01-01"), county %in% c("Adams", "Bond", "Cook"))) + 
  geom_line(aes(x=date, y=new_cases, colour=county)) +
  theme_bw()




data_illinois_smoothed <- data_illinois %>%
  dplyr::arrange(date) %>% 
  dplyr::group_by(county) %>% 
  dplyr::mutate(death_07da = zoo::rollmean(new_deaths, k = 7, fill = NA),
                cases_07da = zoo::rollmean(new_cases, k = 7, fill = NA),
                death_14da = zoo::rollmean(new_deaths, k = 14, fill = NA),
                cases_14da = zoo::rollmean(new_cases, k = 14, fill = NA))

data_illinois_smoothed <- data_illinois_smoothed %>%
  mutate(cases_14da_per_100k = cases_14da/pop * 1e5,
         cases_7da_per_100k = cases_07da/pop * 1e5,
         death_14da_per_100k = death_14da/pop * 1e5,
         death_7da_per_100k = death_07da/pop * 1e5)

ggplot(data_illinois_smoothed %>% filter(county %in% c("Adams", "Bond", "Cook",
                                                       "DuPage",
                                                       "McHenry"))) + 
  geom_line(aes(x=date, y=cases_14da_per_100k, colour=county)) +
  theme_bw()


#### So the data is preprocessed.
#### WE now need to make predictors for each of the counties --- predictors of grow
data_illinois_smoothed = data_illinois_smoothed %>%
  group_by(county) %>%
  mutate(delta_cases_per_day = cases_14da_per_100k - lag(cases_14da_per_100k, k=1, order_by=date),
         delta_cases_per_week = cases_14da_per_100k - lag(cases_14da_per_100k, k=7, order_by=date)
         ) %>%
  mutate(mean_delta_cases_03 = zoo::rollmean(delta_cases_per_day, k = 3, fill = 0, order_by=date),
         mean_delta_cases_05 = zoo::rollmean(delta_cases_per_day, k = 5, fill = 0, order_by=date),
         mean_delta_cases_07 = zoo::rollmean(delta_cases_per_day, k = 7, fill = 0, order_by=date),
         mean_delta_cases_14 = zoo::rollmean(delta_cases_per_day, k = 14, fill = 0, order_by=date),
  ) 

data_illinois_smoothed = data_illinois_smoothed %>%
  group_by(county) %>%
  mutate(av_increase_prev3 = lag(mean_delta_cases_07, k=3, order_by=date),
         av_increase_prev5 = lag(mean_delta_cases_07, k=5, order_by=date),
         av_increase_prev5 = lag(mean_delta_cases_07, k=7, order_by=date),
         av_increase_prev14 = lag(mean_delta_cases_07, k=14, order_by=date),
         av_increase_prev21 = lag(mean_delta_cases_07, k=21, order_by=date),
         av_increase_prev28 = lag(mean_delta_cases_07, k=28, order_by=date),
         
  )






ggplot(data_illinois_smoothed %>% filter(county %in% c("Adams", "Bond", "Cook",
                                                       "DuPage",
                                                       "McHenry")),
       aes(x=av_increase_prev28, y=mean_delta_cases_03)) + 
  geom_point(aes(colour=county)) +
  geom_smooth(method = lm)+
  theme_bw()

ggplot(data_illinois_smoothed,
       aes(x=av_increase_prev28, y=mean_delta_cases_05 )) + 
  geom_point() +
  geom_smooth(method = lm)+
  theme_bw()



#### Try to explain the variability


X = data_illinois_smoothed %>% 
  filter(county %in% c("Adams", "Brown", "Hancock", "Pike County",
                       "Schuyler County",
                       "Lewis", "Marion"))

XX =  X %>% select(mean_delta_cases_05,
                   date, county, )
XXX =  X %>% select(av_increase_prev28,
                   date, county)

X = merge(XX %>% pivot_wider(id_cols = c(date), values_from = c(mean_delta_cases_05), 
                   names_from = c(county), names_prefix="mean_delta_cases_"),
      XXX %>% pivot_wider(id_cols = c(date), values_from = c(av_increase_prev28), 
                         names_from = c(county), names_prefix="three_weeks_earlier_"))
      


dates = seq(from=as.Date("2020-07-01"), to=as.Date("2022-09-01"), by = 14)

reg  = lm( mean_delta_cases_Adams ~three_weeks_earlier_Adams, data= X %>% filter(date %in% dates)) 









data_illinois22 =  data2022 %>% filter(state=='Illinois')
data_illinois21 =  data2021 %>% filter(state=='Illinois')
data_illinois20 =  data2020 %>% filter(state=='Illinois')
data_illinois = rbind(data_illinois22, data_illinois21, data_illinois20)
data_illinois = data_illinois %>%
  mutate(year=year(ymd(date))) %>%
  mutate(year=ifelse(year>2021, 2021, year))  
### join with population on fips
data_illinois = merge(data_illinois, population_data %>% select(fips,Value, year ), 
                      by.x=c('fips', 'year'), by.y=c('fips','year' ),
                      all.x=TRUE,all.y=FALSE)



ggplot(data_illinois %>% filter( fips %in% 17001:17019)) + 
  geom_line(aes(x=date, y=cases, colour=fips)) +
  theme_bw() ### these are the cumulative counts

data_illinois = data_illinois %>%
  group_by(county, date) %>%
  summarize(cases = sum(cases),
            deaths = sum(deaths),
            pop = sum(Value)) %>%
  mutate(new_cases =  cases - lag(cases, k=1, order_by=date),
         new_deaths =  deaths - lag(deaths, k=1, order_by=date))

data_illinois[which(data_illinois["new_cases"]<0), "new_cases"] = 0
data_illinois[which(data_illinois["new_deaths"]<0), "new_deaths"] = 0

ggplot(data_illinois %>% filter( date>ymd("2022-01-01"), county %in% c("Adams", "Bond", "Cook"))) + 
  geom_line(aes(x=date, y=new_cases, colour=county)) +
  theme_bw()




data_illinois_smoothed <- data_illinois %>%
  dplyr::arrange(date) %>% 
  dplyr::group_by(county) %>% 
  dplyr::mutate(death_07da = zoo::rollmean(new_deaths, k = 7, fill = NA),
                cases_07da = zoo::rollmean(new_cases, k = 7, fill = NA),
                death_14da = zoo::rollmean(new_deaths, k = 14, fill = NA),
                cases_14da = zoo::rollmean(new_cases, k = 14, fill = NA))

data_illinois_smoothed <- data_illinois_smoothed %>%
  mutate(cases_14da_per_100k = cases_14da/pop * 1e5,
         cases_7da_per_100k = cases_07da/pop * 1e5,
         death_14da_per_100k = death_14da/pop * 1e5,
         death_7da_per_100k = death_07da/pop * 1e5)

ggplot(data_illinois_smoothed %>% filter(county %in% c("Adams", "Bond", "Cook",
                                                       "DuPage",
                                                       "McHenry"))) + 
  geom_line(aes(x=date, y=cases_14da_per_100k, colour=county)) +
  theme_bw()


#### So the data is preprocessed.
#### WE now need to make predictors for each of the counties --- predictors of grow
data_illinois_smoothed = data_illinois_smoothed %>%
  group_by(county) %>%
  mutate(delta_cases_per_day = cases_14da_per_100k - lag(cases_14da_per_100k, k=1, order_by=date),
         delta_cases_per_week = cases_14da_per_100k - lag(cases_14da_per_100k, k=7, order_by=date)
  ) %>%
  mutate(mean_delta_cases_03 = zoo::rollmean(delta_cases_per_day, k = 3, fill = 0, order_by=date),
         mean_delta_cases_05 = zoo::rollmean(delta_cases_per_day, k = 5, fill = 0, order_by=date),
         mean_delta_cases_07 = zoo::rollmean(delta_cases_per_day, k = 7, fill = 0, order_by=date),
         mean_delta_cases_14 = zoo::rollmean(delta_cases_per_day, k = 14, fill = 0, order_by=date),
  ) 

data_illinois_smoothed = data_illinois_smoothed %>%
  group_by(county) %>%
  mutate(av_increase_prev3 = lag(mean_delta_cases_07, k=3, order_by=date),
         av_increase_prev5 = lag(mean_delta_cases_07, k=5, order_by=date),
         av_increase_prev5 = lag(mean_delta_cases_07, k=7, order_by=date),
         av_increase_prev14 = lag(mean_delta_cases_07, k=14, order_by=date),
         av_increase_prev21 = lag(mean_delta_cases_07, k=21, order_by=date),
         av_increase_prev28 = lag(mean_delta_cases_07, k=28, order_by=date),
         
  )






ggplot(data_illinois_smoothed %>% filter(county %in% c("Adams", "Bond", "Cook",
                                                       "DuPage",
                                                       "McHenry")),
       aes(x=av_increase_prev28, y=mean_delta_cases_03)) + 
  geom_point(aes(colour=county)) +
  geom_smooth(method = lm)+
  theme_bw()

ggplot(data_illinois_smoothed,
       aes(x=av_increase_prev28, y=mean_delta_cases_05 )) + 
  geom_point() +
  geom_smooth(method = lm)+
  theme_bw()



#### Try to explain the variability


X = data_illinois_smoothed %>% 
  filter(county %in% c("Adams", "Brown", "Hancock", "Pike County",
                       "Schuyler County",
                       "Lewis", "Marion"))

XX =  X %>% select(mean_delta_cases_05,
                   date, county, )
XXX =  X %>% select(av_increase_prev28,
                    date, county)

X = merge(XX %>% pivot_wider(id_cols = c(date), values_from = c(mean_delta_cases_05), 
                             names_from = c(county), names_prefix="mean_delta_cases_"),
          XXX %>% pivot_wider(id_cols = c(date), values_from = c(av_increase_prev28), 
                              names_from = c(county), names_prefix="three_weeks_earlier_"))



dates = seq(from=as.Date("2020-07-01"), to=as.Date("2022-09-01"), by = 14)

reg  = lm( mean_delta_cases_Adams ~three_weeks_earlier_Adams, data= X %>% filter(date %in% dates)) 







##### TEXAS
STATE = "Texas"
data_state22 =  data2022 %>% filter(state==STATE)
data_state21 =  data2021 %>% filter(state==STATE)
data_state20 =  data2020 %>% filter(state==STATE)
data_state = rbind(data_state22, data_state21, data_state20)
data_state = data_state %>%
  mutate(year=year(ymd(date))) %>%
  mutate(year=ifelse(year>2021, 2021, year))  
### join with population on fips
data_state = merge(data_state, population_data %>% select(fips,Value, year ), 
                      by.x=c('fips', 'year'), by.y=c('fips','year' ),
                      all.x=TRUE,all.y=FALSE)


data_state$fips
ggplot(data_state %>% filter( fips %in% unique(data_state$fips)[1:10])) + 
  geom_line(aes(x=date, y=cases, colour=fips)) +
  theme_bw() ### these are the cumulative counts

data_state = data_state %>%
  group_by(county, date) %>%
  summarize(cases = sum(cases),
            deaths = sum(deaths),
            pop = sum(Value)) %>%
  mutate(new_cases =  cases - lag(cases, k=1, order_by=date),
         new_deaths =  deaths - lag(deaths, k=1, order_by=date))

data_state[which(data_state["new_cases"]<0), "new_cases"] = 0
data_state[which(data_state["new_deaths"]<0), "new_deaths"] = 0

ggplot(data_state %>%filter(county %in% unique(data_state$county)[1:10]))+ 
  geom_line(aes(x=date, y=new_cases, colour=county)) +
  theme_bw()

data_state_smoothed <- data_state %>%
  dplyr::arrange(date) %>% 
  dplyr::group_by(county) %>% 
  dplyr::mutate(death_07da = zoo::rollmean(new_deaths, k = 7, fill = NA),
                cases_07da = zoo::rollmean(new_cases, k = 7, fill = NA),
                death_14da = zoo::rollmean(new_deaths, k = 14, fill = NA),
                cases_14da = zoo::rollmean(new_cases, k = 14, fill = NA))

data_state_smoothed <- data_state_smoothed %>%
  mutate(cases_14da_per_100k = cases_14da/pop * 1e5,
         cases_7da_per_100k = cases_07da/pop * 1e5,
         death_14da_per_100k = death_14da/pop * 1e5,
         death_7da_per_100k = death_07da/pop * 1e5)

ggplot(data_state_smoothed %>% filter(county %in% unique(data_state$county)[1:4])) + 
  geom_line(aes(x=date, y=cases_7da_per_100k, colour=county)) +
  theme_bw()


#### So the data is preprocessed.
#### WE now need to make predictors for each of the counties --- predictors of grow
data_state_smoothed = data_state_smoothed %>%
  group_by(county) %>%
  mutate(diff_new_cases= cases_7da_per_100k - lag(cases_7da_per_100k, k=1, order_by=date),
         diff_new_cases_week = cases_7da_per_100k - lag(cases_7da_per_100k, k=7, order_by=date)
  ) %>%
  mutate(mean_delta_cases_03 = zoo::rollmean(diff_new_cases, k = 3, fill = 0, order_by=date),
         mean_delta_cases_05 = zoo::rollmean(diff_new_cases, k = 5, fill = 0, order_by=date),
         mean_delta_cases_07 = zoo::rollmean(diff_new_cases, k = 7, fill = 0, order_by=date),
         mean_delta_cases_14 = zoo::rollmean(diff_new_cases, k = 14, fill = 0, order_by=date),
  ) 

data_state_smoothed = data_state_smoothed %>%
  group_by(county) %>%
  mutate(av_increase_prev3 = lag(mean_delta_cases_07, k=3, order_by=date),
         av_increase_prev5 = lag(mean_delta_cases_07, k=5, order_by=date),
         av_increase_prev5 = lag(mean_delta_cases_07, k=7, order_by=date),
         av_increase_prev14 = lag(mean_delta_cases_07, k=14, order_by=date),
         av_increase_prev21 = lag(mean_delta_cases_07, k=21, order_by=date),
         av_increase_prev28 = lag(mean_delta_cases_07, k=28, order_by=date),
         
  )




subset_county = sample(unique(data_state$county),4)

ggplot(data_state_smoothed %>% filter(county %in% subset_county),
       aes(x=av_increase_prev28, y=mean_delta_cases_07)) + 
  geom_point(aes(colour=county)) +
  geom_smooth(method = lm)+
  theme_bw()

#### Try to explain the variability


X = data_state_smoothed 
XX =  X %>% select(mean_delta_cases_07,
                   date, county, )
XXX =  X %>% select(av_increase_prev28,
                    date, county)

X = merge(XX %>% filter(county == 'Borden') %>% pivot_wider(id_cols = c(date), values_from = c(mean_delta_cases_07), 
                             names_from = c(county), names_prefix="mean_delta_cases_"),
          XXX %>% pivot_wider(id_cols = c(date), values_from = c(av_increase_prev28), 
                              names_from = c(county), names_prefix="three_weeks_earlier_"))


write_csv(X %>% 
            replace(is.na(.), 0), file = "~/Downloads/full_X.csv")
write_csv( X %>% filter(date %in% train_dates) %>% 
             replace(is.na(.), 0), file = "~/Downloads/X_train.csv")
write_csv( X %>% filter(date %in% test_dates) %>% 
             replace(is.na(.), 0), file = "~/Downloads/X_test.csv")
dates = seq(from=as.Date("2020-07-01"), to=as.Date("2022-09-01"), by = 14)

train_dates = dates[1:40]
test_dates = dates[41:length(dates)]

reg  = lm( mean_delta_cases_Borden ~ . ,
           data= X %>% filter(date %in% train_dates) %>% 
             replace(is.na(.), 0)) 

t = predict.lm(reg, newdata = X %>% 
                 replace(is.na(.), 0))
summary(reg)
temp = X  %>% 
  replace(is.na(.), 0)
temp$pred = t

ggplot(temp, 
       aes(x=date,y=pred)) +
  geom_point() + 
  geom_point(aes(y=mean_delta_cases_Borden), colour="red") +
  theme_bw()

t = predict.lm(reg, newdata = X %>% filter(date %in% test_dates) %>% 
              replace(is.na(.), 0))
summary(reg)
temp = X %>% filter(date %in% test_dates) %>% 
  replace(is.na(.), 0)
temp$pred = t
ggplot(temp, aes(x=date,y=pred)) +
  geom_point()


ggplot(data_state_smoothed %>% filter(county %in% c('Dawson',
                                                    'Garza',
                                                    'Howard',
                                                    'Lynn',
                                                    'Martin',
                                                    'Mitchell',
                                                    'Scurry', 'Borden')),
       aes(x=av_increase_prev28, y=mean_delta_cases_07)) + 
  geom_point(aes(colour=county)) +
  geom_smooth(method = lm)+
  theme_bw()

ggplot(data_state_smoothed %>% filter(county %in% c('Dawson',
                                                    'Garza',
                                                    'Howard',
                                                    'Lynn',
                                                    'Martin',
                                                    'Mitchell',
                                                    'Scurry', 'Borden')),
       aes(x=av_increase_prev28, y=mean_delta_cases_07)) + 
  geom_point(aes(colour=county)) +
  geom_smooth(method = lm)+
  theme_bw()







##### We could also try at the fips level

reg

