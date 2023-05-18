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
from sklearn.preprocessing import PolynomialFeatures

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
    
    def data_import_feature_engineering(data):
        
        #Importing in-function libraries
        import calendar
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import PolynomialFeatures

        #Importing data:
        df = pd.read_csv(data)

        #df = input_df.copy()

        #Drop unnamed column
        df = df.drop('Unnamed: 0',axis=1)   
        #Rename time and turn into dt format        
        df = df.rename(columns={'time': 'time_stamp'})
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])   
        #Create weekday predictor:
        df['weekday'] = df['time_stamp'].dt.weekday
        #Extract the month from the "date" column and create a new column "month"
        df['month'] = df['time_stamp'].dt.month
        #Extract the hour from the date column and create a new column called hour
        df['hour'] = df['time_stamp'].dt.hour
            #Map month dictionary
        month_names = {
            1: 'January',
            2: 'February',
            3: 'March',
            4: 'April',
            5: 'May',
            6: 'June',
            7: 'July',
            8: 'August',
            9: 'September',
            10: 'October',
            11: 'November',
            12: 'December'
        }
            #Map the month numbers to month names
        df['month'] = df['month'].map(month_names)
            # Define a dictionary to map months to seasons
        seasons = {
            'January': 'Winter',
            'February': 'Winter',
            'March': 'Spring',
            'April': 'Spring',
            'May': 'Spring',
            'June': 'Summer',
            'July': 'Summer',
            'August': 'Summer',
            'September': 'Fall',
            'October': 'Fall',
            'November': 'Fall',
            'December': 'Winter'
        }
            # Map the months to seasons
        df['season'] = df['month'].map(seasons)
            #Define day dictionary
        days = {
             0: 'Monday',
             1: 'Tuesday',
             2: 'Wednesday',
             3: 'Thursday',
             4: 'Friday',
             5: 'Saturday',
             6: 'Sunday'
        }     
            #Map days to day names 
        df['weekday'] = df['weekday'].map(days)
            #Adjusting function for training dataset
        cols_to_remove = ['time_stamp']

        #Remove time and load shortfall cols
        remove_cols = df[cols_to_remove]
        remove = df[cols_to_remove].columns 
            #Temp separate remove datafrom above
        df = df.drop(remove, axis=1)   
            #Change Valencia wind reg to numerical (remove text)
        df['Valencia_wind_deg'] = df['Valencia_wind_deg'].str.replace('level_','').astype(int)
            #Change Saville pressure to numerical (remove test)
        df['Seville_pressure'] = df['Seville_pressure'].str.replace('sp','').astype(int) 

        #Determne numerical cols and impute mean for missing values
        numerical_cols  = df.select_dtypes(include=['number'])
        def mean_impute(col):
             col_ave = col.mean()
             return col.fillna(col_ave)    
        numerical_cols = numerical_cols.apply(mean_impute) 
            #Determne numerical cols and impute mean for missing values
        categorical_cols  = df.select_dtypes(exclude=['number']) 
        def mean_impute(col):
             col_ave = col.mean()
             return col.fillna(col_ave)    
        numerical_cols = numerical_cols.apply(mean_impute) 
            #Impute dummy variables:
        categorical_cols = pd.get_dummies(categorical_cols,dtype=bool,drop_first=True)

        #Reassamble database
        df = pd.concat([remove_cols,numerical_cols,categorical_cols],axis=1).drop('time_stamp',axis=1)
        return df
    
    df = data_import_feature_engineering("df_test.csv")

    def train_scaler_and_polyconverter(df):

        #Create instance of scaler and scale X data
        scaler = StandardScaler()
        df = scaler.fit_transform(df)

        return df

    predict_vector = train_scaler_and_polyconverter(df)
    
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
