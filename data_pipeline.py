import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#function to load the data adding the column 'year' to facilitate the splitting
def load_data(path):

  df = pd.read_excel(path)

  df['year'] = df['Date'] //10000  #adding 'year' column

  return df

#function to split the data into train, validation and test sets
def split_data(df):

  train_df = df[df['year'] <= 2016].reset_index(drop=True) #primi 17 anni

  valid_df = df[(df['year'] >= 2017) & (df['year'] <= 2019)].reset_index(drop=True) # secondi 3 anni

  test_df = df[df['year'] >= 2020].reset_index(drop=True) # ultimi 3 anni

  return train_df, valid_df, test_df

#function that remove leaked features and clip the target variable by replacing negative values with 0
def clean_data(df):

  df = df.drop(columns=['ET0', 'Water demand']) #removing 'ET0' and 'Water demand' columns

  df['Amount of irrigation'] = df['Amount of irrigation'].clip(lower=0) #replacing negative values with 0 in 'Amount of irrigation' column

  return df

#funzione di normalizzazione (prende i 3 dataset già puliti e restituisce i 3 dataset normalizzati)
def normalize(train_df_clean, valid_df_clean, test_df_clean):
  columns = train_df_clean.columns.drop(['Date', 'year']) #nomi delle colonne escluse Date e year (non eliminate)

  #calcolo il minimo e il massimo per ogni colonna del train senza tenere conto delle due colonne
  train_df_min = train_df_clean[columns].min()
  train_df_max = train_df_clean[columns].max()
  #(questi valori mi serviranno anche per la normalizzazione inversa)

  #operazione di normalizzazione min-max sui 3 dataframe, usando i minimi e massimi del train per evitare data leakage
  train_df_clean[columns] = (train_df_clean[columns] - train_df_min) / (train_df_max - train_df_min)
  valid_df_clean[columns] = (valid_df_clean[columns] - train_df_min) / (train_df_max - train_df_min)
  test_df_clean[columns] = (test_df_clean[columns] - train_df_min) / (train_df_max - train_df_min)

  return train_df_clean, valid_df_clean, test_df_clean, train_df_min, train_df_max