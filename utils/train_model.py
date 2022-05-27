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



# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')
train_1 = train

# Data Preprocessing
train_1['Valencia_pressure'] = train_1['Valencia_pressure'].fillna(train['Valencia_pressure'].mode()[0])
train_1['time'] = pd.to_datetime(train['time'])

#Transform the Seville_pressure column
if train_1.Valencia_wind_deg.dtypes == 'O':
    train_1['Valencia_wind_deg'] = train_1['Valencia_wind_deg'].str.extract('(\d+)')  #extract the numbers from the string
    train_1['Valencia_wind_deg'] = pd.to_numeric(train_1['Valencia_wind_deg'])        #next, transform from object datatype to numeric

#Transform the Seville_pressure column
if train.Seville_pressure.dtypes == 'O' :
    train_1['Seville_pressure'] = train_1['Seville_pressure'].str.extract('(\d+)')      #extract the numbers from the string
    train_1['Seville_pressure'] = pd.to_numeric(train_1['Seville_pressure'])            #next, transform from object datatype to numeric


# Transform Time feature
train_1['Year'] = train_1['time'].dt.year    # year
train_1['Day'] = train_1['time'].dt.day      # Day
train_1['Month'] = train_1['time'].dt.month  # month
train_1['hour'] = train_1['time'].dt.hour    # hour

# Rearrange the features
# columsn_list = ['Year','Month','Day','hour'] + list(train_1.columns[1:-4])
# train_sub = train_1[columsn_list]
train_sub = train_1


to_drop_list = ['Unnamed: 0','time']
drop_list = [col for col in train_sub.columns if  col in to_drop_list ]
train_sub = train_sub.drop(drop_list,axis=1)
X_columns = [col for col in train_sub.columns if  col != 'load_shortfall_3h' ]
# for col in X_columns:
#     print(col)
#     print('\t\t')

y_train = train_sub[['load_shortfall_3h']]
X_train = train_sub[X_columns]

# print(X_train.columns)





# y_train = train[['load_shortfall_3h']]
# X_train = train[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
# print(y_train)
# print(X_train)
# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
# save_path = '../assets/trained-models/load_shortfall_simple_lm_regression.pkl'
save_path = '../assets/trained-models/jm1_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))
