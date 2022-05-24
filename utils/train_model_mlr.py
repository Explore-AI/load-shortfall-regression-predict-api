# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Fetch training data and preprocess for modeling
train = pd.read_csv(r'C:\Users\Roger Arendse\Desktop\EDSA\Technical\5. Advanced Regression\1. Predict\load-shortfall-regression-predict-api\utils\data\df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Seville_humidity','Bilbao_rain_1h','Valencia_wind_speed' ]]

# Fit model
lr = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
#lr = LinearRegression(normalize=True)

print ("Training Model...")
lr.fit(X_train, y_train)

# Pickle model for use within our API
model_save_path = r'C:\Users\Roger Arendse\Desktop\EDSA\Technical\5. Advanced Regression\1. Predict\load-shortfall-regression-predict-api\assets\mlr_model.pkl'
with open(model_save_path,'wb') as file:
    pickle.dump(lr, file)
