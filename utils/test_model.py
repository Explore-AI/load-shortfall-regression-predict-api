# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

    # Fetch training data and preprocess for modeling
feature_vector_df = df_test = pd.read_csv('./data/df_test.csv')



    # Data Preprocessing
df_test['Valencia_pressure'] = df_test['Valencia_pressure'].fillna(df_test['Valencia_pressure'].mode()[0])
df_test['time'] = pd.to_datetime(df_test['time'])

    #Transform the Seville_pressure column
if df_test.Valencia_wind_deg.dtypes == 'O':
    df_test['Valencia_wind_deg'] = df_test['Valencia_wind_deg'].str.extract('(\d+)')  #extract the numbers from the string
    df_test['Valencia_wind_deg'] = pd.to_numeric(df_test['Valencia_wind_deg'])        #next, transform from object datatype to numeric

    #Transform the Seville_pressure column
if df_test.Seville_pressure.dtypes == 'O' :
    df_test['Seville_pressure'] = df_test['Seville_pressure'].str.extract('(\d+)')      #extract the numbers from the string
    df_test['Seville_pressure'] = pd.to_numeric(df_test['Seville_pressure'])            #next, transform from object datatype to numeric


# Transform Time feature
df_test['Year'] = df_test['time'].dt.year    # year
df_test['Day'] = df_test['time'].dt.day      # Day
df_test['Month'] = df_test['time'].dt.month  # month
df_test['hour'] = df_test['time'].dt.hour    # hour

# Rearrange the features
columsn_list = ['Year','Month','Day','hour'] + list(df_test.columns[1:-4])
test_sub = df_test[columsn_list]
test_sub = df_test


to_drop_list = ['Unnamed: 0','time']
drop_list = [col for col in test_sub.columns if  col in to_drop_list ]
test_sub = test_sub.drop(drop_list,axis=1)
X_columns = [col for col in test_sub.columns if  col != 'load_shortfall_3h' ]

X_test = test_sub[X_columns]
    
# predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
# predict_vector = X_test

print(X_test)
