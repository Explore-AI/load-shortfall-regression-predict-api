import pandas as pd
import numpy as np

feature_vector_df = pd.read_csv('df_train.csv')



feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'])

feature_vector_df = feature_vector_df.drop(['Unnamed: 0', 'time'], axis=1)

feature_vector_df['Valencia_pressure'] = feature_vector_df['Valencia_pressure'].fillna(value=feature_vector_df['Valencia_pressure'].mean())
feature_vector_df['Valencia_wind_deg'] = feature_vector_df['Valencia_wind_deg'].str.extract('(\d+)')
feature_vector_df['Valencia_wind_deg'] =pd.to_numeric(feature_vector_df['Valencia_wind_deg'])
feature_vector_df['Seville_pressure'] = feature_vector_df['Seville_pressure'].str.extract('(\d+)')
feature_vector_df['Seville_pressure'] =pd.to_numeric(feature_vector_df['Seville_pressure'])
