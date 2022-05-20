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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

PARAMETER_CONSTANT = 13

# Fetch training data and preprocess for modeling
print("pickle me this")
train = pd.read_csv('./data/df_train.csv')

# y_train = train[['load_shortfall_3h']]
# X_train = train[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]

print("pickle me this")
y2 = train['load_shortfall_3h']
X2 = train('load_shortfall_3h', axis= 1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3332, random_state= PARAMETER_CONSTANT)
# Fit model
r_forest_model_1 = RandomForestRegressor(n_estimators = 100, random_state = PARAMETER_CONSTANT)
# lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
r_forest_model_1.fit(X2,y2)
r_forest_model_1_pred = r_forest_model_1.predict(X2)
# lm_regression.fit(y2, X2)

# Pickle model for use within our API
save_path = '../assets/trained-models/load_shortfall_simple_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(r_forest_model_1_pred   , open(save_path,'wb'))
# pickle.dump(lm_regression, open(save_path,'wb'))
