"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Libraries for data loading, data manipulation and data visulisation
import pandas as pd
import numpy as np
# Libraries for data preparation and model building
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
# Import train/test splitting function from sklearn to split the data into training and testing data
from sklearn.model_selection import train_test_split
# Import the ridge regression module from sklearn
from sklearn.linear_model import Ridge
#Importing the linear model from Sklearn
from sklearn.linear_model import LinearRegression
# Import metrics module
from sklearn import metrics
# Setting global constants to ensure notebook results are reproducible
#PARAMETER_CONSTANT = ###

import pickle
# Fetch training data and preprocess for modeling
df_train = pd.read_csv('./data/df_train.csv', index_col=0)

#converting the time column to datetime data type
df_train['time'] = pd.to_datetime(df_train['time'])


#creating the temporal features 
df_train['Day_of_Week'] = df_train['time'].dt.dayofweek
df_train['Week_of_Year'] = df_train['time'].dt.weekofyear
df_train['Day_of_Year'] = df_train['time'].dt.dayofyear
df_train['Month_of_Year'] = pd.DatetimeIndex(df_train['time']).month #Actual Month
df_train['Year'] = pd.DatetimeIndex(df_train['time']).year #Actual Year
df_train['Day_of_Month'] = pd.DatetimeIndex(df_train['time']).day #Day of month
df_train['Hour_of_Day'] = pd.DatetimeIndex(df_train['time']).hour #Hour of day
df_train['Hour_of_Year'] = (df_train['time'].dt.dayofyear )* 24 + df_train['time'].dt.hour #Hour of year -1
df_train['Hour_of_Week'] = (df_train['time'].dt.dayofweek ) * 24 +  df_train['time'].dt.hour #Hour of week


#selecting and creating the dummy variables for the categorical features
cat_var = df_train[['Valencia_wind_deg', 'Seville_pressure']]
cat_var_dum = pd.get_dummies(cat_var, drop_first=True)
#Now let's copy the training data to a new dataframe
df2 = df_train.copy()
#dropping the column with Object data type - a technique to improve our model
df2 = df2.drop(['time', 'Valencia_wind_deg', 'Seville_pressure'], axis=1)
# Creating a dataframe of our response or target variable
response_y = df2['load_shortfall_3h']

df2 = df2.drop('load_shortfall_3h', axis=1)

# create scaler object
scaler = StandardScaler()
# convert the scaled predictor values into a dataframe
df2 = pd.DataFrame(scaler.fit_transform(df2),columns = df2.columns)

# feature engineering on existing features
# This also imputes missing values of the features
imputer = KNNImputer(n_neighbors=6)
df2 = pd.DataFrame(imputer.fit_transform(df2),columns = df2.columns)

df2 = pd.concat([df2, cat_var_dum], axis=1)
column_titles = [i for i in df2.columns]
df2 = df2.reindex(columns = column_titles)
# Drop All temp_max and temp_min of all the cities to avoid Multicollinearity(here we need to show how they are corelated before dropping them)
df2 = df2.drop(['Seville_temp_max','Valencia_temp_max','Barcelona_temp_max','Madrid_temp_max','Bilbao_temp_max','Seville_temp_min','Valencia_temp_min','Barcelona_temp_min','Madrid_temp_min','Bilbao_temp_min'],axis=1)


# Now, let's drop the redundant time variables(we need to show why they are redundant)
df2 = df2.drop(['Week_of_Year','Month_of_Year', 'Day_of_Week', 'Year', 'Hour_of_Year'], axis=1)
import xgboost as xgb
data_dmatrix = xgb.DMatrix(data=df2,label=response_y)
#creating the features and target variables
X = df2
y = response_y
# split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=50)
# create one or more ML models

print ("Training Model...")
#create Xgboost
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.7,n_estimators = 120, max_depth = 5, subsample = 0.7, learning_rate = 0.1)


# Fit the Ml model
xg_reg.fit(x_train, y_train)
# Pickle model for use within our API
save_path = '../assets/trained-models/team_jm3.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(xg_reg, open(save_path,'wb'))
