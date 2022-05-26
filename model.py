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
    print(df_train['Valencia_pressure'].mean())
1012.0514065222828
#Filling in the missing values with the mean
df_train['Valencia_pressure'].fillna(df_train['Valencia_pressure'].mean(), inplace = True)
#converting Valencia_wind_deg and Seville_pressure columns from categorical to numerical datatypes.

df_train['Valencia_wind_deg'] = df_train['Valencia_wind_deg'].str.extract('(\d+)').astype('int64')
df_train['Seville_pressure'] = df_train['Seville_pressure'].str.extract('(\d+)').astype('int64')
The next step is to engineer new features from the time column

#Engineering New Features ( i.e Desampling the Time) to further expand our training data set

df_train['Year']  = df_train['time'].astype('datetime64').dt.year
df_train['Month_of_year']  = df_train['time'].astype('datetime64').dt.month
df_train['Week_of_year'] = df_train['time'].astype('datetime64').dt.weekofyear
df_train['Day_of_year']  = df_train['time'].astype('datetime64').dt.dayofyear
df_train['Day_of_month']  = df_train['time'].astype('datetime64').dt.day
df_train['Day_of_week'] = df_train['time'].astype('datetime64').dt.dayofweek
df_train['Hour_of_week'] = ((df_train['time'].astype('datetime64').dt.dayofweek) * 24 + 24) - (24 - df_train['time'].astype('datetime64').dt.hour)
df_train['Hour_of_day']  = df_train['time'].astype('datetime64').dt.hour
Let us have a look at the correlation(s) between our newly created temporal features

Time_df = df_train.iloc[:,[-8,-7,-6,-5,-4,-3,-2,-1]]
plt.figure(figsize=[10,6])
sns.heatmap(Time_df.corr(),annot=True )
<AxesSubplot:>

Looking at our heatmap tells us that we have high Multicollinearity present in our new features. The features involved are -

Week of the year. Day of the year. Month of the year. Day of the week. Hour of the week. We would have to drop one of the features that have high correlation with each other.

Alongside dropping these features mentioned above, we would also be dropping the time and Unnamed column.

df_train = df_train.drop(columns=['Week_of_year','Day_of_year','Hour_of_week', 'Unnamed: 0','time'])
plt.figure(figsize=[35,15])
sns.heatmap(df_train.corr(),annot=True )
<AxesSubplot:>

Just as we mentioned in our EDA, we noticed the presence of high correlations between the predictor columns and also possible outliers.

Here, we would have to drop these columns to improve the performance of our model and reduce any possibility of overfitting in our model. Let us check if this approach corresponds with our feature selection. Using SelectKBest and Chi2 to perform Feature Selection.

## Splitting our data into dependent Variable and Independent Variable
X = df_train.drop(columns = 'load_shortfall_3h')
y = df_train['load_shortfall_3h'].astype('int')
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Features', 'Score']
new_X = featureScores.sort_values('Score',ascending=False).head(40)
new_X.head(40) #To get the most important features based on their score 
Features	Score
18	Barcelona_pressure	1.189344e+09
9	Bilbao_wind_deg	4.574064e+05
8	Seville_clouds_all	3.049398e+05
11	Barcelona_wind_deg	2.920143e+05
12	Madrid_clouds_all	2.862344e+05
6	Bilbao_clouds_all	1.705834e+05
32	Bilbao_weather_id	1.307308e+05
24	Barcelona_weather_id	7.121392e+04
5	Madrid_humidity	7.087652e+04
17	Bilbao_snow_3h	6.812971e+04
4	Seville_humidity	5.699050e+04
23	Madrid_weather_id	5.445955e+04
26	Seville_weather_id	4.703123e+04
34	Valencia_humidity	3.980066e+04
48	Day_of_month	3.443358e+04
50	Hour_of_day	3.167767e+04
15	Seville_pressure	2.687804e+04
14	Barcelona_rain_1h	2.171411e+04
3	Valencia_wind_speed	1.601889e+04
47	Month_of_year	1.293213e+04
1	Valencia_wind_deg	1.104442e+04
7	Bilbao_wind_speed	1.092892e+04
0	Madrid_wind_speed	1.017244e+04
49	Day_of_week	9.265478e+03
13	Seville_wind_speed	8.132635e+03
10	Barcelona_wind_speed	8.016649e+03
2	Bilbao_rain_1h	7.544582e+03
16	Seville_rain_1h	5.397681e+03
20	Madrid_rain_1h	4.226512e+03
29	Madrid_pressure	3.436256e+03
22	Valencia_snow_3h	3.110384e+03
37	Madrid_temp_max	2.281817e+03
44	Madrid_temp	2.106589e+03
45	Madrid_temp_min	2.054920e+03
28	Seville_temp_max	1.847097e+03
43	Seville_temp_min	1.589866e+03
33	Seville_temp	1.483057e+03
30	Valencia_temp_max	1.365686e+03
36	Barcelona_temp_max	1.260724e+03
31	Valencia_temp	1.229799e+03
This result backups our claim, were we saw in the heatmap multicollinearity between features, and from our feature selection, we can see those features as having the lowest significance in our data.

Dropping Outliers We have one more thing to do, which is to remove possible outliers. Also, we will select the important features for our model thus dropping others having multicollinearity

X = X[['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h',
       'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
       'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
       'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h',
       'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
       'Barcelona_rain_3h', 'Madrid_weather_id',
       'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
       'Valencia_pressure', 'Bilbao_weather_id', 
        'Valencia_humidity', 'Year', 'Month_of_year', 'Day_of_month', 'Day_of_week', 'Hour_of_day']]
plt.figure(figsize=[20,10])
sns.heatmap(X.corr(),annot=True )
<AxesSubplot:>

We have been able to remove the collinearity seen in previous heatmaps and also selected specific features to train our model with

Feature Scaling Lastly, before we carry out modeling, it is important to scale our data. As we saw during the EDA, we noticed how some columns(features) had values that were out of range when we compared their mean, max and standard deviation. This can result to bias in the model during decision making, thus it is important to convert all the column values to a certain range/scale.

What is Feature Scaling? Feature scaling is the process of normalising the range of features in a dataset. Real-world datasets often contain features that are varying in degrees of magnitude, range and units. Therefore, in order for machine learning models to interpret these features on the same scale, we need to perform feature scaling.

In this project, we will be carrying out Standard Scaling, becasue of it's robustness to outliers

# Create standardization object
scaler = StandardScaler()
# Save standardized features into new variable
#"""
#We used a fit transform method, which first fits in the standardscaler and then transforms the data """
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,columns=X.columns)
X_scaled.head()

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
    predict_vector = feature_vector_df[['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h',
       'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
       'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
       'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h',
       'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
       'Barcelona_rain_3h', 'Madrid_weather_id',
       'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
       'Valencia_pressure', 'Bilbao_weather_id', 
        'Valencia_humidity', 'Year', 'Month_of_year', 'Day_of_month', 'Day_of_week', 'Hour_of_day']]
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
