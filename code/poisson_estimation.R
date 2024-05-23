

# install libraries if they are not already installed
if (!requireNamespace("fixest", quietly = TRUE)) {
  install.packages("fixest")
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}


library(fixest)
library(dplyr)


df = read.csv('../data/individual_data.csv')


# create variables for gender 
df[df$gender == 'female', 'gender'] = 1
df[df$gender == 'male', 'gender'] = 0


us = df[(df$country == 'us'),]
uk = df[(df$country == 'uk'),]


us = us[us$issue != 'international_affairs',]


## create function to fit poisson model 

fit_poisson_male <- function(data) {
  # fit poisson model
  model <- fepois(issue_tweets ~ log_ratio_male_mip * gender + vote_share | name + survey_date_int + issue,
  data = data)
  # return model
  return(model)
}

fit_poisson_female <- function(data) {
  # fit poisson model
  model <- fepois(issue_tweets ~ log_ratio_female_mip * gender + vote_share | name + survey_date_int + issue + party,
  data = data)
  # return model
  return(model)
}



us_m1 = fit_poisson_male(us)
us_m2 = fit_poisson_female(us)
uk_m1 = fit_poisson_male(uk)
uk_m2 = fit_poisson_female(uk)


## print results 
etable(us_m1, us_m2, uk_m1, uk_m2, vcov = ~ name + survey_date_int, tex = TRUE,
       dict=c(log_ratio_female_mip="Women's Salience", 
              log_ratio_male_mip="Men's Salience", 
              vote_share =  'Vote Share', 
              issue_tweets = 'Issue Tweets', 
              gender1 = 'Women Rep.',
              survey_date_int = 'Time',
              issue = 'Issue', 
              party = 'Party',
              name = 'Legislator'
              ))




