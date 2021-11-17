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
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]

    predict_vector = feature_vector_df
    # ------------------------------------------------------------------------

    X=feature_vector_df[['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h',
       'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
       'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
       'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h',
       'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
       'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id',
       'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
       'Valencia_pressure', 'Seville_temp_max', 'Madrid_pressure',
       'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id',
       'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min',
       'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp',
       'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min',
       'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp',
       'Madrid_temp_min']]




    df=X.copy()

    X['Valencia_pressure'] = X['Valencia_pressure'].fillna(X.Valencia_pressure.mode()[0])

    #create dummy variables  for the Seville_pressure and also for the Valencia_wind_deg in the train data
    X['Seville_pressure']=X.Seville_pressure.map({'sp25':25, 'sp23':23, 'sp24':24, 'sp21':21, 'sp16':16, 'sp9':9, 'sp15':15, 'sp19':19, 'sp22':22, 'sp11':11,
    'sp8':8, 'sp4':4, 'sp6':6, 'sp13':13, 'sp17':17, 'sp20':20, 'sp18':18, 'sp14':14, 'sp12':12, 'sp5':5, 'sp10':10,
    'sp7':7, 'sp3':3, 'sp2':2, 'sp1':1})

    X['Valencia_wind_deg']=X.Valencia_wind_deg.map({'level_5':5, 'level_10':10, 'level_9':9, 'level_8':8, 'level_7':7, 'level_6':6, 'level_4':4,
    'level_3':3, 'level_1':1, 'level_2':2})

    #standardization
    from sklearn.preprocessing import StandardScaler

    # create scaler object
    scaler = StandardScaler()

    # create scaled version of the predictors (there is no need to scale the response)
    test_scaled = scaler.fit_transform(X)

    # convert the scaled predictor values into a dataframe
    X = pd.DataFrame(test_scaled,columns=X.columns)

    return X


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
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
