import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Seville_wind_speed', 'Barcelona_wind_speed', 'Bilbao_rain_1h']]

# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/final_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))
