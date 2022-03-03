import pandas as pd
import numpy as np

df = pd.read_csv('df_train.csv')

df_clean = df
df_clean['Valencia_pressure'] = df_clean['Valencia_pressure'].fillna(value=df['Valencia_pressure'].mean())
df_clean['time'] = pd.to_datetime(df_clean['time'])
df_clean['Valencia_wind_deg'] = df_clean['Valencia_wind_deg'].str.extract('(\d+)')
df_clean['Valencia_wind_deg'] =pd.to_numeric(df_clean['Valencia_wind_deg'])
df_clean['Seville_pressure'] = df_clean['Seville_pressure'].str.extract('(\d+)')
df_clean['Seville_pressure'] =pd.to_numeric(df_clean['Seville_pressure'])
df_clean = df_clean.drop(['Unnamed: 0', 'time'], axis=1)