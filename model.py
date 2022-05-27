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
    # feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------
    # train = train_df = pd.DataFrame.from_dict([feature_vector_dict])

    train = train_df = pd.DataFrame.from_dict(feature_vector_dict)
    train_1 = train

    # Data Preprocessing
    train_1['Valencia_pressure'] = train_1['Valencia_pressure'].fillna(train_1['Valencia_pressure'].mode()[0])
    train_1['time'] = pd.to_datetime(train_1['time'])

    # Transform the Seville_pressure column
    if train_1.Valencia_wind_deg.dtypes == 'O':
        train_1['Valencia_wind_deg'] = train_1['Valencia_wind_deg'].str.extract(
            '(\d+)')  # extract the numbers from the string
        train_1['Valencia_wind_deg'] = pd.to_numeric(
            train_1['Valencia_wind_deg'])  # next, transform from object datatype to numeric

    # Transform the Seville_pressure column
    if train.Seville_pressure.dtypes == 'O':
        train_1['Seville_pressure'] = train_1['Seville_pressure'].str.extract(
            '(\d+)')  # extract the numbers from the string
        train_1['Seville_pressure'] = pd.to_numeric(
            train_1['Seville_pressure'])  # next, transform from object datatype to numeric

    # Transform Time feature
    train_1['Year'] = train_1['time'].dt.year  # year
    train_1['Day'] = train_1['time'].dt.day  # Day
    train_1['Month'] = train_1['time'].dt.month  # month
    train_1['hour'] = train_1['time'].dt.hour  # hour

    train_sub = train_1

    to_drop_list = ['Unnamed: 0', 'time']
    drop_list = [col for col in train_sub.columns if col in to_drop_list]
    train_sub = train_sub.drop(drop_list, axis=1)
    X_columns = [col for col in train_sub.columns if col != 'load_shortfall_3h']

    predict_vector = train_sub[X_columns]

    # ----------- Replace this code with your own preprocessing steps --------
    # predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
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
    print(prep_data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
