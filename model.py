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
    
    
    #############################starts here ..........#####################
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    df = pd.DataFrame.from_dict([feature_vector_dict])

    
    df['time'] = pd.to_datetime(df['time'])
    df['Day_of_Year'] = df['time'].dt.dayofyear
    df['Day_of_Month'] = pd.DatetimeIndex(df['time']).day #Day of month
    df['Hour_of_Day'] = pd.DatetimeIndex(df['time']).hour #Hour of day
    df['Hour_of_Week'] = (df['time'].dt.dayofweek ) * 24 +  df['time'].dt.hour #Hour of week
    
    #giving a new name for valencia
    valencia = df['Valencia_wind_deg'][0]
    Seville = df['Seville_pressure'][0]
    valencia = ('Valencia_wind_deg'+'_'+valencia)
    Seville = ('Seville_pressure'+'_'+Seville)
    cat_var = [valencia, Seville]

    expected = ['Valencia_wind_deg_level_10', 'Valencia_wind_deg_level_2',
           'Valencia_wind_deg_level_3', 'Valencia_wind_deg_level_4',
           'Valencia_wind_deg_level_5', 'Valencia_wind_deg_level_6',
           'Valencia_wind_deg_level_7', 'Valencia_wind_deg_level_8',
           'Valencia_wind_deg_level_9', 'Seville_pressure_sp10',
           'Seville_pressure_sp11', 'Seville_pressure_sp12',
           'Seville_pressure_sp13', 'Seville_pressure_sp14',
           'Seville_pressure_sp15', 'Seville_pressure_sp16',
           'Seville_pressure_sp17', 'Seville_pressure_sp18',
           'Seville_pressure_sp19', 'Seville_pressure_sp2',
           'Seville_pressure_sp20', 'Seville_pressure_sp21',
           'Seville_pressure_sp22', 'Seville_pressure_sp23',
           'Seville_pressure_sp24', 'Seville_pressure_sp25',
           'Seville_pressure_sp3', 'Seville_pressure_sp4', 'Seville_pressure_sp5',
           'Seville_pressure_sp6', 'Seville_pressure_sp7', 'Seville_pressure_sp8',
           'Seville_pressure_sp9']

    expectedval = []
    for i in expected:
        if i in cat_var:
            i = 1
        else:
            i = 0
        expectedval.append(i)
    

    final_dum = pd.DataFrame([0],columns = ['init'])

    n = 0
    while n < len(expected):
        final_dum[expected[n]] = expectedval[n]
        n = n+1
    df.Valencia_pressure = 1018
     

    final_dum = final_dum.drop('init', axis =1)
    df = df.drop(['time', 'Valencia_wind_deg', 'Seville_pressure'], axis=1)

    df = pd.concat([df, final_dum], axis=1)
    # Drop All temp_max and temp_min of all the cities to avoid Multicollinearity(here we need to show how they are corelated before dropping them)
    df =     df.drop(['Seville_temp_max','Valencia_temp_max','Barcelona_temp_max','Madrid_temp_max','Bilbao_temp_max','Seville_temp_min','Valencia_temp_min','Barcelona_temp_min','Madrid_temp_min','Bilbao_temp_min'],axis=1)
    
    predict_vector = df

    ###############################  end here #####################################
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
