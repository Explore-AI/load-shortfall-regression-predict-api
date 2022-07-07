"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
import bz2
import _pickle as cPickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

PARAMETER_CONSTANT = 13

# Fetch training data and preprocess for modeling
print("pickle me this")
train = pd.read_csv('./data/df_train.csv')

train['time'] = pd.to_datetime(train['time'])

train['time'] = pd.to_datetime(train['time'], format = '%Y-%m-%d %H:%M:%S')

train['year'] = train['time'].dt.year     # year value is arbitrary where power is concerned
train['month'] = train['time'].dt.month   # power varies per month depending on season
train['day'] =train['time'].dt.day       # power varies depending on day of the week
train['hour'] =train['time'].dt.hour     # power varies depending on the time of the day

train[['month', 'day', 'hour']] = train[['month', 'day', 'hour']].astype('int64')

df_date = [i for i in train.columns if i != 'load_shortfall_3h'] + ['load_shortfall_3h']
train= train.reindex(columns=df_date)

    #Extracting the numeric on our data but datatype still object
train['Seville_pressure'] = train['Seville_pressure'].str.extract('(\d+)')
train['Valencia_wind_deg'] = train['Valencia_wind_deg'].str.extract('(\d+)')

    #converting object into numeric data type
train['Seville_pressure'] = pd.to_numeric(train['Seville_pressure'])
train['Valencia_wind_deg'] = pd.to_numeric(train['Valencia_wind_deg'])

X = train.drop(['load_shortfall_3h', 'time'], axis=1)
y = train['load_shortfall_3h']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_standardise = pd.DataFrame(X_scaled, columns=X.columns)
X_standardise = pd.DataFrame(X_scaled, columns=X.columns)



train = train.drop(['Seville_pressure', 'Valencia_wind_deg', 'time' , 'Valencia_pressure'], axis=1)

X2 = train.drop('load_shortfall_3h', axis= 1)
y2 = train['load_shortfall_3h']

# Fit model
# rfr = RandomForestRegressor(n_estimators =200, max_depth=None, max_features='auto', bootstrap=True, random_state =PARAMETER_CONSTANT)

lm= LinearRegression(normalize=True)
print ("Training Model...")

lm.fit(X2,y2)


# Pickle model for use within our API
save_path = '../assets/trained-models/load_shortfall_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")


# with bz2.BZ2File(save_path+'.pbz2', 'w') as f: 
#     cPickle.dump(rfr, f)
# pickle.dump(xgb , open(save_path,'wb'))
pickle.dump(lm, open(save_path,'wb'))
