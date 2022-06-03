# Import dependencies
import requests
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=6)
# Load data from file to send as an API POST request.
# We prepare a DataFrame with the public test set + riders data
# from the Kaggle challenge.
#test = pd.read_csv('./data/df_test.csv')


#am using this part to ensure that our data is to the corect formating

df_test = pd.read_csv('./data/df_test.csv', index_col=0)
df_return = df_test.copy()


def clean_data(df_test):
    test_df = df_test.copy()
    test_df['time'] = pd.to_datetime(test_df['time'])
    test_df['Day_of_Week'] = test_df['time'].dt.dayofweek
    test_df['Week_of_Year'] = test_df['time'].dt.weekofyear
    test_df['Day_of_Year'] = test_df['time'].dt.dayofyear
    test_df['Month_of_Year'] = pd.DatetimeIndex(test_df['time']).month #Actual Month
    test_df['Year'] = pd.DatetimeIndex(test_df['time']).year #Actual Year
    test_df['Day_of_Month'] = pd.DatetimeIndex(test_df['time']).day #Day of month
    test_df['Hour_of_Day'] = pd.DatetimeIndex(test_df['time']).hour #Hour of day
    test_df['Hour_of_Year'] = (test_df['time'].dt.dayofyear )* 24 + test_df['time'].dt.hour #Hour of year -1
    test_df['Hour_of_Week'] = (test_df['time'].dt.dayofweek ) * 24 +  test_df['time'].dt.hour #Hour of week
    #df_train = df_train.drop('time', axis=1)

    test_time = test_df[['time']]
    test_time = test_time.reset_index().drop(["index"], axis=1)
    #drop the categorical columns 
    test_df = test_df.drop(['time', 'Valencia_wind_deg', 'Seville_pressure' ], axis=1)

    #sacling the dataFrame using the StandarScalar
    test_df = pd.DataFrame(scaler.fit_transform(test_df), columns = test_df.columns)
    #Imputing the missing values using Knn Imputer
    test_df = pd.DataFrame(imputer.fit_transform(test_df), columns = test_df.columns)

    #creating dummies for test categorical variables
    cat_var = df_test[['Valencia_wind_deg', 'Seville_pressure']]
    test_dummies = pd.get_dummies(cat_var, drop_first=True)

    #resetting the index and droping the index column
    test_dummies = test_dummies.reset_index().drop(["index"], axis=1)

    #concatenating the test dataframe and the test_dummies dataframe
    test_df = pd.concat([test_df, test_dummies], axis=1)

    # DROPPING REDUNDANT TIME VARIABLES
    test_df = test_df.drop(['Week_of_Year','Month_of_Year', 'Day_of_Week', 'Year', 'Hour_of_Year'], axis=1)

    # Drop All temp_max and temp_min all the cities to avoid Multicollinearity
    test_df = test_df.drop(['Seville_temp_max','Valencia_temp_max','Barcelona_temp_max','Madrid_temp_max','Bilbao_temp_max','Seville_temp_min','Valencia_temp_min','Barcelona_temp_min','Madrid_temp_min','Bilbao_temp_min'],axis=1)
    return test_df

#this is the end of my code and data is returned

test = clean_data(df_test)




# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = test.iloc[1].to_json()

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API on a remote machine
# replace the URL below with its public IP:

# url = 'http://{public-ip-address-of-remote-machine}:5000/api_v0.1'
url = 'http://127.0.0.1:5000/api_v0.1'

# Perform the POST request.
print(f"Sending POST request to web server API at: {url}")
print("")
print(f"Querying API with the following data: \n {df_return.iloc[1].to_list()}")
print("")
# Here `api_response` represents the response we get from our API
api_response = requests.post(url, json=feature_vector_json)

# Display the prediction result
print("Received POST response:")
print("*"*50)
print(f"API prediction result: {api_response.json()}")
print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
print("*"*50)
