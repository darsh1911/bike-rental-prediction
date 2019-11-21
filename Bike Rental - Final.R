rm(list = ls())

#load libraries
library(data.table)
library(caret)
library(lubridate)
library(DataExplorer)
library(outliers)
library(MLmetrics)
library(rpart)
library(randomForest)
library(mda)

#setting the working directory
setwd('/Users/darsh/Documents/Data Science/Bike Rental Project/R/')

#load dataset
data = read.csv('day.csv')
data_key = data$instant
data_target = data$cnt

#Check for missing values
anyNA(data)

dataset = data
dataset$instant = NULL
dataset$cnt = NULL

year = year(ymd(dataset$dteday))
year = as.factor(year)
dataset$yr = NULL

month = month(ymd(dataset$dteday), label = TRUE, abbr = FALSE)
month = as.factor(month)
dataset$mnth = NULL

rownames(dataset) = dataset$dteday
dataset$dteday = NULL

dataset = cbind(year, month, dataset)

dataset$season = as.factor(dataset$season)
levels(dataset$season) = c('Spring', 'Summer', 'Fall', 'Winter')

dataset$holiday = as.factor(dataset$holiday)

dataset$weekday = as.factor(dataset$weekday)
levels(dataset$weekday) = c('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday')

dataset$workingday = as.factor(dataset$workingday)
levels(dataset$workingday) = c('Weekend', 'Weekday')

dataset$holiday = as.factor(dataset$holiday)
levels(dataset$holiday) = c('Non-Holiday', 'Holiday')

dataset$weathersit = as.factor(dataset$weathersit)
levels(dataset$weathersit) = c('Clear', 'Cloudy/Mist', 'Rain/Snow/Fog', 'Heavy Rain/Snow/Fog')

#Generate a correlation plot to analyse the correlation between the features
plot_correlation(dataset, type = 'continuous', title = 'Correlation Plot')

#Dropping casual and registered variables as they are highly correlated to cnt
dataset$casual = NULL
dataset$registered = NULL

#Dropping atemp as it is highly correlated to temp
dataset$atemp = NULL

#Isolating continuous and discrete features for analysis
xy = split_columns(dataset)

#Generating bar plots for discrete and histograms for continuous features
plot_bar(cbind(xy$discrete, data_target), with = 'data_target', ncol = 4)
plot_histogram(xy$continuous, ncol = 3)

#Replacing outliers from continuous variables with median values
xy$continuous = rm.outlier(xy$continuous, fill = TRUE, median = TRUE)

discrete = sapply(xy$discrete, as.numeric)

dataset = cbind(discrete, xy$continuous, data_target)
setnames(dataset, 'data_target', 'cnt')

#Splitting the dataset into train and test
set.seed(16)
split = createDataPartition(dataset$cnt, p = 0.75, list = FALSE)
training = dataset[split,]
validation = dataset[-split,]

#Evaluating algorithms

#Build models
#Decision tree model
model.rpart = rpart(cnt ~., data = training ,method ='anova')
score = predict(model.rpart, validation)
MAPE(score, validation$cnt)

#RandomForest model
model.rf = randomForest(cnt ~., data =training, importance = TRUE, ntree = 50)
score = predict(model.rf, validation)
MAPE(score, validation$cnt)

#Multivariate adaptiveregression splines (MARS) model
model.mars = mars(training[,-c(11)], training[,11])
score = predict(model.mars, validation[,-c(11)])
MAPE(score, validation$cnt)