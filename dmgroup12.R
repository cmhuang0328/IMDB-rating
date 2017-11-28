# IMDB movie dataset data mining project Group 12
library(dplyr)
library(ggplot2)
library(ggrepel)
library(stringr)
library(Hmisc)
library(stringr)
library(magrittr)
library(randomForest)
library (caret) 
library(gbm)
library(neuralnet)
library(nnet)
library (h2o)
library(xgboost)
library(VIM)
library(mice)
library(Hmisc)

# Path setup
system("ls ../input")
movie = read.csv("movie_metadata.csv", header = TRUE)

# Exploration Data Analysis
nrow(movie)
# 5043 movies
names(movie)
# 28 attributes
summary(movie)
# each attributes their min, median, mean, max, NA's
missmap(movie)

# Movie Poster
ggplot(data = movie) + geom_point(mapping = aes(x = facenumber_in_poster, y = imdb_score)) + geom_smooth(mapping = aes(x = facenumber_in_poster, y = imdb_score), se = F)

marginplot(movie[,c("imdb_score", "gross")])
marginplot(movie[,c("imdb_score", "budget")])

marginplot(movie[,c("gross", "imdb_score")])
marginplot(movie[,c("budget", "imdb_score")])
marginplot(movie[,c("aspect_ratio", "imdb_score")])
marginplot(movie[,c("title_year", "imdb_score")])
marginplot(movie[,c("director_facebook_likes", "imdb_score")])
marginplot(movie[,c("actor_1_facebook_likes", "imdb_score")])
pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(data,2,pMiss)

# Data preprocessing
movie_mis<- movie
# Fix Movie Names
movie_mis$movie_title <- gsub("Â","",movie_mis$movie_title)
movie_mis$movie_title <- str_trim(movie_mis$movie_title)
# Fix Duplicate Movie Title
movie_mis <- movie_mis[!duplicated(movie_mis$movie_title),]
# Fix Facebook Likes NAs with mean
summary(movie$director_facebook_likes)
summary(movie$actor_1_facebook_likes)
summary(movie$actor_2_facebook_likes)
summary(movie$actor_3_facebook_likes)
movie_mis$director_facebook_likes <- with(movie_mis, impute(director_facebook_likes, mean))
movie_mis$actor_1_facebook_likes <- with(movie_mis, impute(actor_1_facebook_likes, mean))
movie_mis$actor_2_facebook_likes <- with(movie_mis, impute(actor_2_facebook_likes, mean))
movie_mis$actor_3_facebook_likes <- with(movie_mis, impute(actor_3_facebook_likes, mean))
# Remove Useless Column
movie_mis$movie_imdb_link<- NULL
# Remove instances which have at least one NA variable
movie_mis <- movie_mis[complete.cases(movie_mis), ]
# Zoom in on the most experienced stars
# We choose 10 starring roles as an arbitrary cut-off
actors <- movie_data %>% group_by(actor_1_name) %>% 
  summarise(count=n(), avg_score=mean(imdb_score)) %>% 
  arrange(desc(count))
top_actors <- actors %>% filter(count > 10)
ggplot(top_actors, aes(count, avg_score)) + geom_point(alpha=1/3) + 
    geom_text_repel(aes(label=actor_1_name), size=2)

# Outlier
source("outlier.R")
outlierKD(movie, facenumber_in_poster)
# Remove instances which have at least one NA variable
movie_mis <- movie_mis[complete.cases(movie_mis), ]

# Feature Selection

# Analyzing use numerical datas to do corrplot
nums <- sapply(movie_mis, is.numeric) 
movie_cor <- movie_mis[,nums]  %>% correlate() %>% focus(imdb_score)
movie_cor_all <- movie_mis[, c("imdb_score", unique(movie_cor$rowname))] %>% cor()
corrplot(movie_cor_all, method= "circle", main = "Correlation Matrix")

select_columns_final <- c('imdb_score', 'num_user_for_reviews', 'num_critic_for_reviews', 'duration', 
                         'director_facebook_likes', 'movie_facebook_likes', 'cast_total_facebook_likes',
                         'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
                         'gross', 'num_voted_users', 'facenumber_in_poster', 'budget', 
                         'title_year', 'aspect_ratio')

movie_data <- movie_mis[,select_columns_final]

# training and test data buildup
inTrain <- createDataPartition (movie_data[,'imdb_score'], p=0.8, list= FALSE ) 
training <- movie_data[inTrain,] 
test <- movie_data[-inTrain,]

# Linear Regression
movie.linear = lm(imdb_score~num_voted_users+duration+num_critic_for_reviews+num_user_for_reviews+movie_facebook_likes+gross, data=training)
# make prediction on the validation data
summary(movie.linear)
movie.linear
predicted.linear = predict(movie.linear, test)
# calculate the Test RMSE
rmse.li<- sqrt(mean((test$imdb_score - predicted.linear)^2))
rmse.li

# Random Forest
inTrain <- createDataPartition (movie_data[,'imdb_score'], p=0.8, list= FALSE ) 
training <- movie_data[inTrain,] 
test <- movie_data[-inTrain,] 
movie.rf <- randomForest(imdb_score~.,data=training, mtry=6,importance=TRUE)
summary(movie.rf)
movie.rf1<- randomForest(imdb_score~~num_voted_users+duration+num_critic_for_reviews+num_user_for_reviews+movie_facebook_likes+gross,data=training, mtry=6,importance=TRUE)
predicted.rf1 <- predict(movie.rf1,newdata=test)
mse.rf1<- (sum((test$imdb_score - predicted.rf1)^2, na.rm=T)/nrow(movie_data))^0.5
mse.rf1
predicted.rf <- predict(movie.rf,newdata=test)
# calculate the Test RMSE
rmse.rf<- sqrt(mean((test$imdb_score - predicted.rf)^2))
rmse.rf


# Boosting
# we can use the same data as we do for random forest
movie.boost<- gbm(imdb_score~.,data=training,distribution="gaussian",n.trees=5000,interaction.depth=4)
summary(movie.boost)
predicted.boost<- predict(movie.boost,newdata=test,n.trees=5000)
# calculate the Test RMSE
rmse.boost<- sqrt(mean((test$imdb_score - predicted.boost)^2))
rmse.boost

# neutral network
movie.nn<- neuralnet( imdb_score ~ num_user_for_reviews+num_critic_for_reviews+duration+
                               director_facebook_likes+movie_facebook_likes+cast_total_facebook_likes+
                               actor_1_facebook_likes+actor_2_facebook_likes+actor_3_facebook_likes+
                               gross+num_voted_users+facenumber_in_poster+budget+
                               title_year+aspect_ratio, 
                       data=training, hidden=c(5))
plot(movie.nn)
predicted.nn<- compute(movie.nn, test[,2:16])$net.result

model <- train(form = imdb_score ~ num_user_for_reviews+num_critic_for_reviews+duration+
                               director_facebook_likes+movie_facebook_likes+cast_total_facebook_likes+
                               actor_1_facebook_likes+actor_2_facebook_likes+actor_3_facebook_likes+
                               gross+num_voted_users+facenumber_in_poster+budget+
                               title_year+aspect_ratio, data = training, method = "neuralnet", 
               tuneGrid = expand.grid(.layer1 = c(5:6), .layer2 = c(0), .layer3 = c(0)), learningrate = 0.1)
model
# calculate the Test RMSE
rmse.nn<- sqrt(mean((test$imdb_score - predicted.nn)^2))
rmse.nn
