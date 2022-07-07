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
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import pickle
import bz2
import _pickle as cPickle
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
    
    # ------------------------------------------------------------------------

    # remove missing values/ features
    # mode = pd.concat([feature_vector_df.Valencia_pressure]).mode()
    # feature_vector_df.Valencia_pressure.fillna(mode[0] , inplace=True)

    # df_test1 = feature_vector_df.copy()
   
    # mode = pd.concat([ feature_vector_df.Valencia_pressure]).mode()
    # print("sdfgh" , mode[0])
    # feature_vector_df.Valencia_pressure.fillna(mode[0] , inplace=True)

    # create new features
    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'])

    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'], format = '%Y-%m-%d %H:%M:%S')

    feature_vector_df['year'] = feature_vector_df['time'].dt.year     # year value is arbitrary where power is concerned
    feature_vector_df['month'] = feature_vector_df['time'].dt.month   # power varies per month depending on season
    feature_vector_df['day'] = feature_vector_df['time'].dt.day       # power varies depending on day of the week
    feature_vector_df['hour'] = feature_vector_df['time'].dt.hour     # power varies depending on the time of the day

    feature_vector_df[['month', 'day', 'hour']] = feature_vector_df[['month', 'day', 'hour']].astype('int64')

    df_date = [i for i in feature_vector_df.columns if i != 'load_shortfall_3h'] + ['load_shortfall_3h']
    feature_vector_df= feature_vector_df.reindex(columns=df_date)
    print("shape here )))))))))))=========" , feature_vector_df.shape)
    #Extracting the numeric on our data but datatype still object
    feature_vector_df['Seville_pressure'] = feature_vector_df['Seville_pressure'].str.extract('(\d+)')
    feature_vector_df['Valencia_wind_deg'] = feature_vector_df['Valencia_wind_deg'].str.extract('(\d+)')

    #converting object into numeric data type
    feature_vector_df['Seville_pressure'] = pd.to_numeric(feature_vector_df['Seville_pressure'])
    feature_vector_df['Valencia_wind_deg'] = pd.to_numeric(feature_vector_df['Valencia_wind_deg'])

    mode = pd.concat([feature_vector_df.Valencia_pressure]).mode()
    # print(mode)
    

    X = feature_vector_df.drop(['load_shortfall_3h', 'time'], axis=1)
    y = feature_vector_df['load_shortfall_3h']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_standardise = pd.DataFrame(X_scaled, columns=X.columns)
    X_standardise = pd.DataFrame(X_scaled, columns=X.columns)
  

    feature_vector_df = feature_vector_df.drop(['Seville_pressure','load_shortfall_3h' ,'Valencia_wind_deg', 'time', 'Valencia_pressure'], axis=1)
    # feature_vector_df= feature_vector_df.drop(['Valencia_pressure'] , axis=1)
    predict_vector = feature_vector_df

    return feature_vector_df

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
    # data = bz2.BZ2File(path_to_model + '.pbz2', 'rb')
    # data = cPickle.load(data)
    
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
