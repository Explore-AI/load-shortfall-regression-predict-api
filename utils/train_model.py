"""
    GM 5 Random forest model
"""

# Dependencies
import pickle

# Libraries for data loading, data manipulation and data visulisation
import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for data preparation and model building
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from collections import Counter
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR,SVC, LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import r2_score as r2
#########################################################################
#Feature engineering and data import function (combined to facilitate pipeline)
def data_import_feature_engineering(data):
     
     #Importing in-function libraries
     import calendar
     import pandas as pd

     #Importing data:
     df = pd.read_csv(data)

     #df = input_df.copy()
     
     #Drop unnamed column
     df = df.drop('Unnamed: 0',axis=1)   

     #Rename time and turn into dt format        
     df = df.rename(columns={'time': 'time_stamp'})

     df['time_stamp'] = pd.to_datetime(df['time_stamp'])   

     #Create weekday predictor:
     df['weekday'] = df['time_stamp'].dt.weekday
     #Extract the month from the "date" column and create a new column "month"
     df['month'] = df['time_stamp'].dt.month
     #Extract the hour from the date column and create a new column called hour
     df['hour'] = df['time_stamp'].dt.hour

     #Map month dictionary
     month_names = {
         1: 'January',
         2: 'February',
         3: 'March',
         4: 'April',
         5: 'May',
         6: 'June',
         7: 'July',
         8: 'August',
         9: 'September',
         10: 'October',
         11: 'November',
         12: 'December'
     }

     #Map the month numbers to month names
     df['month'] = df['month'].map(month_names)

     # Define a dictionary to map months to seasons
     seasons = {
         'January': 'Winter',
         'February': 'Winter',
         'March': 'Spring',
         'April': 'Spring',
         'May': 'Spring',
         'June': 'Summer',
         'July': 'Summer',
         'August': 'Summer',
         'September': 'Fall',
         'October': 'Fall',
         'November': 'Fall',
         'December': 'Winter'
     }

     # Map the months to seasons
     df['season'] = df['month'].map(seasons)

     #Define day dictionary
     days = {
          0: 'Monday',
          1: 'Tuesday',
          2: 'Wednesday',
          3: 'Thursday',
          4: 'Friday',
          5: 'Saturday',
          6: 'Sunday'
     }     

     #Map days to day names 
     df['weekday'] = df['weekday'].map(days)

     #Adjusting function for training dataset
     if 'load_shortfall_3h' not in df.columns:
          cols_to_remove = ['time_stamp']
     else:
          cols_to_remove = ['time_stamp','load_shortfall_3h']

     
     #Remove time and load shortfall cols
     remove_cols = df[cols_to_remove]
     remove = df[cols_to_remove].columns 

     #Temp separate remove datafrom above
     df = df.drop(remove, axis=1)   

     #Change Valencia wind reg to numerical (remove text)
     df['Valencia_wind_deg'] = df['Valencia_wind_deg'].str.replace('level_','').astype(int)

     #Change Saville pressure to numerical (remove test)
     df['Seville_pressure'] = df['Seville_pressure'].str.replace('sp','').astype(int) 
     
     #Determne numerical cols and impute mean for missing values
     numerical_cols  = df.select_dtypes(include=['number'])
     def mean_impute(col):
          col_ave = col.mean()
          return col.fillna(col_ave)    
     numerical_cols = numerical_cols.apply(mean_impute) 

     #Determne numerical cols and impute mean for missing values
     categorical_cols  = df.select_dtypes(exclude=['number']) 
     def mean_impute(col):
          col_ave = col.mean()
          return col.fillna(col_ave)    
     numerical_cols = numerical_cols.apply(mean_impute) 

     #Impute dummy variables:
     categorical_cols = pd.get_dummies(categorical_cols,dtype=bool,drop_first=True)
     
     #Reassamble database
     df = pd.concat([remove_cols,numerical_cols,categorical_cols],axis=1).drop('time_stamp',axis=1)

     return df
#########################################################################

#Run feature engineering and data import function
df = data_import_feature_engineering("df_train.csv")

# split data
X = df.drop('load_shortfall_3h',axis =1) 
y = df['load_shortfall_3h'] 


#########################################################################
def train_scaler_and_polyconverter(X, y):
    #Split data up into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  

    #Create instance of scaler and scale X data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_scaler_and_polyconverter(X,y)
#########################################################################

#Create instance of SVM model
RF_model = RandomForestRegressor(n_estimators=2)

#Fit grid model to training data
RF_model.fit(X_train, y_train)

#Predict y test values
RF_model_preds = RF_model.predict(X_test)

#Generate MSE and calculate RMSE and Rsq
RMSE_RF_model = round(np.sqrt(mean_squared_error(y_test,RF_model_preds)))

#Print results
print('RMSE_RF_model',RMSE_RF_model)





#########################################################################


# Pickle model for use within our API
save_path = 'C:/Users/nassa/OneDrive/Documents/Github/load-shortfall-regression-predict-api/assets/trained-models/load_shortfall_RF.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(RF_model, open(save_path,'wb'))

