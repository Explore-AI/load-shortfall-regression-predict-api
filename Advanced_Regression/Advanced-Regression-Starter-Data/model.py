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
    data = data.drop(['Unnamed: 0'],axis=1) #remove the unnamed col
    data = data.drop([col for col in data if 'id' in col], axis ='columns')
    data = data.drop(['Seville_pressure','Valencia_wind_deg'],axis='columns')

    #create new features
    temp_features = [col for col in data if 'temp' in col and 'av_spain_temp' not in col]
    data['av_spain_temp'] = data[temp_features].mean(axis=1)
    data = data.drop(temp_features, axis='columns')

    pressure_features = [col for col in data if 'pressure' in col and 'av_spain_pressure' not in col]
    data['av_spain_pressure'] = data[pressure_features].mean(axis=1)
    data = data.drop(pressure_features, axis='columns')

    rain_features = [col for col in data if 'rain' in col and 'av_spain_rain' not in col]
    data['av_spain_rain'] = data[rain_features].mean(axis=1)
    data = data.drop(rain_features, axis='columns')

    wind_features = [col for col in data if 'wind' in col and 'av_spain_wind' not in col]
    data['av_spain_wind'] = data[wind_features].mean(axis=1)
    data = data.drop(wind_features, axis='columns')

    snow_features = [col for col in data if 'snow' in col and 'av_spain_snow' not in col]
    data['av_spain_snow'] = data[snow_features].mean(axis=1)
    data = data.drop(snow_features, axis='columns')

    clouds_features = [col for col in data if 'cloud' in col and 'av_spain_clouds' not in col]
    data['av_spain_clouds'] = data[clouds_features].mean(axis=1)
    data = data.drop(clouds_features, axis='columns')

    time =  pd.to_datetime(data['time'])
    data['time'] = time
    data['Day'] = data['time'].dt.day
    data['month'] = data['time'].dt.month
    data['hour'] = data['time'].dt.hour
    data = data.drop('time', axis='columns')

    # create scaler object
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(X_scaled,columns=data.columns)
    #after normalizing, we can fill nan with zero
    data.fillna(0)


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
    predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    # ------------------------------------------------------------------------

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
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
