# Generalization of NBA Salary Prediction Model  

## Introduction  
With ever increasing team payrolls, owners can usually afford to pay an underachieving player.  The real issue is in determining the opportunity cost of this:  What could you have gotten for that money instead?  How else could you have improved your roster?  It should come as no surprise then that teams are looking for alternative methods for determining appropriate pay levels for their athletes.  This is where predictive analytics can hopefully shed some needed light on this very expensive problem as laid out below:  
* What should an NBA player be paid based on his performance statistics?  
* Which players are drastically overpaid?  
* Which players are drastically underpaid?  

One of the main focuses of this project was to determine how well our previous model would generalize to data collected from 2010-2019.  Generalization is a term used to describe a model’s ability to react to new data. That is, after being trained on a training set, a model can digest new data and make accurate predictions. A model’s ability to generalize is central to the success of a model.  

## Data Sources Used  
Data was scraped from [basketball-reference.com] and [espn.com].  The data were then combined into a single DataFrame.  

## Technologies Used  
* Python 3+  
* Jupyter Notebook 5.7.8  
* R 3.5  
* Custom web scraper  

## Required Packages  
### Python Packages for Web Scraper  
```python
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
```  
### Python Packages for Model  
```python
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import preprocessing
from collections import defaultdict
from xgboost import XGBRegressor, plot_importance
import seaborn as sns
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
```  
### R Packages  
```R
library(ggplot2)
library(gridExtra)
library(grid)
library(forcats)
library(magrittr)
library(dplyr)
```  

## Analysis Methods Used  
* Linear Regression  
* Extreme Gradient Boosting Modeling  
* Feature Selection  
* Graphic Analysis  
* Web Scraping  
* Exploratory Data Analysis  
* Tableau  

## Model Deployment  
The Extreme Gradient Boosting Model (XGBR) was used on all 10-years worth of data based on model performance metrics established in the previous study.  

## Summary of Results  
| Year | R^2 | RMSE |  
| ---- | ---- | ---- |  
| 2010 | 0.946 | 1,099,627 |  
| 2011 | 0.949 | 1,070,874 |  
| 2012 | 0.957 | 989,150 |  
| 2013 | 0.942 | 1,132,491 |  
| 2014 | 0.946 | 1,174,769 |  
| 2015 | 0.940 | 1,204,874 |  
| 2016 | 0.955 | 1,147,450 |  
| 2017 | 0.940 | 1,641,827 |  
| 2018 | 0.926 | 2,124,853 |  
| 2019 | 0.923 | 2,246,051 |  

The XGBR model provided excellent generalization to our new datasets.   The R^2 values were high and the root mean square error (RMSE) values were considerably lower than what we saw with the original four models in the previous study.  One trend we did start to notice was that the model seems to be decreasing in performance in the last four years.  The R^2 values are decreasing and the RMSE values are increasing.  We are not sure at this point what is causing this, but it is an interesting trend and it grabbed our attention.
