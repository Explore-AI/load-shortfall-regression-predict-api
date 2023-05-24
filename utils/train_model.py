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
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Madrid_wind_speed','Valencia_wind_speed','Bilbao_wind_speed', 
    'Barcelona_wind_speed','Seville_wind_speed','Barcelona_rain_1h',
    'Seville_rain_1h','Bilbao_snow_3h','Seville_rain_3h','Madrid_rain_1h', 
    'Barcelona_rain_3h', 'Valencia_snow_3h','Seville_temp_max','Valencia_temp_max', 
    'Valencia_temp','Seville_temp','Valencia_temp_min', 'Barcelona_temp_max', 
    'Madrid_temp_max','Barcelona_temp', 'Bilbao_temp_min', 'Bilbao_temp',
    'Barcelona_temp_min', 'Bilbao_temp_max', 'Seville_temp_min','Madrid_temp', 'Madrid_temp_min']]

# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/load_shortfall_simple_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))
