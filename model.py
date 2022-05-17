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
    # predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    # ------------------------------------------------------------------------


    feature_vector_df ['time'] = pd.to_datetime(feature_vector_df['time'])

    dum_cols = ['Valencia_wind_deg', 'Seville_pressure']
    df_dummies = pd.get_dummies(feature_vector_df, prefix='dummies', prefix_sep='_', columns=dum_cols, dtype=int, drop_first=True)

    df_dummies.columns = [col.replace(" ","_") for col in df_dummies.columns]



    dum_title = [i for i in df_dummies.columns if i != 'load_shortfall_3h'] + ['load_shortfall_3h']
    df_dummies = df_dummies.reindex(columns=dum_title)

    feature_vector_df  = df_dummies.copy()

    X = feature_vector_df.drop(['load_shortfall_3h', 'time'], axis=1)
    y = feature_vector_df['load_shortfall_3h']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_standardise = pd.DataFrame(X_scaled, columns=X.columns)
    X_standardise.head()

    predict_vector =feature_vector_df

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
    print("here-----"  ,  prep_data )
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    print("here-----"  , prediction[0].tolist())
    return prediction[0].tolist()
