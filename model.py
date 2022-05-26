"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *



def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    print ("here--------" , data)

    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    
    # ------------------------------------------------------------------------

    #replace with means
    feature_vector_df['Valencia_pressure'] = feature_vector_df['Valencia_pressure'].fillna(feature_vector_df['Valencia_pressure'].mean())


    # both Valencia_wind_deg and Seville_pressure dropped
    feature_vector_df.drop('Valencia_wind_deg', axis= 1, inplace= True)
    feature_vector_df.drop('Seville_pressure', axis= 1, inplace= True)


    # create new features
    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'], format = '%Y-%m-%d %H:%M:%S')

    feature_vector_df['month'] = feature_vector_df['time'].dt.month   # power varies per month depending on season
    feature_vector_df['day'] = feature_vector_df['time'].dt.day       # power varies depending on day of the week
    feature_vector_df['hour'] = feature_vector_df['time'].dt.hour 


    # feature_vector_df[['month', 'day', 'hour']] = feature_vector_df[['month', 'day', 'hour']].astype('category')

    # rain_1h_columns = ['Bilbao_rain_1h', 'Barcelona_rain_1h', 'Seville_rain_1h', 'Madrid_rain_1h']
    # Percentage unique values in all rain_1h columns
    # As we can see the unique percentage is less than 1%.
    # Thus only less than 1% of the values is unique.

    # Drop these columns and test models to see if there is a difference.

    # Model performance did increase slightly, thus drop columns.

    # (feature_vector_df[rain_1h_columns].nunique()/ 8763) *100


    # weather_id_columns = ['Madrid_weather_id','Barcelona_weather_id','Seville_weather_id', 'Bilbao_weather_id']

    # Percentage unique values in all weather_id columns
    # As we can see the unique percentage is 4% or less.
    # Thus only 4% or less of the values is unique.

    # Drop these columns and test models to see if there is a difference.

    # Model performance did decrease slightly, Thus weather_id_columns will not be dropped.

    # (feature_vector_df[weather_id_columns].nunique()/ 8763) *100

    #data 2 model 2
    # feature_vector_df1= feature_vector_df.copy()
    # feature_vector_df1 = feature_vector_df.drop(rain_1h_columns, axis=1) 

    predict_vector = feature_vector_df[['month', 'day', 'hour']] = feature_vector_df[['month', 'day', 'hour']].astype('category')
    # predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    # split data

    # y1 = feature_vector_df['load_shortfall_3h']
    # X1 = feature_vector_df.drop('load_shortfall_3h', axis= 1)



    # X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3332, random_state= PARAMETER_CONSTANT)


    # feature_vector_df.drop('Valencia_wind_deg', axis= 1, inplace= True)
    # feature_vector_df.drop('Seville_pressure', axis= 1, inplace= True)




    # feature_vector_df ['time'] = pd.to_datetime(feature_vector_df['time'])

    # dum_cols = ['Valencia_wind_deg', 'Seville_pressure']
    # df_dummies = pd.get_dummies(feature_vector_df, prefix='dummies', prefix_sep='_', columns=dum_cols, dtype=int, drop_first=True)

    # df_dummies.columns = [col.replace(" ","_") for col in df_dummies.columns]



    # dum_title = [i for i in df_dummies.columns if i != 'load_shortfall_3h'] + ['load_shortfall_3h']
    # df_dummies = df_dummies.reindex(columns=dum_title)

    # feature_vector_df  = df_dummies.copy()

    # X = feature_vector_df.drop(['load_shortfall_3h', 'time'], axis=1)
    # y = feature_vector_df['load_shortfall_3h']

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # X_standardise = pd.DataFrame(X_scaled, columns=X.columns)
    # X_standardise.head()

    # predict_vector =feature_vector_df
    print("***********************************get here done" , predict_vector )
    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    print("here-----1")
    # Perform prediction with model and preprocessed data.
    
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    print("here---the end--"  , prediction[0].tolist())
    return prediction[0].tolist()
