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

#funzione di normalizzazione (prende 3 dataset e li normalizza)
def normalize(train_df_, valid_df_, test_df_):

  #set da normalizzare, uso una loro copia per non sovrascrivere i set con i valori reali
  train_df_ = train_df_.copy() 
  valid_df_ = valid_df_.copy()
  test_df_ = test_df_.copy()
  
  columns = train_df_.columns.drop(['Date', 'year']) #nomi delle colonne escluse Date e year (non eliminate)

  #calcolo il minimo e il massimo per ogni colonna del train senza tenere conto delle due colonne
  train_df_min = train_df_[columns].min()
  train_df_max = train_df_[columns].max()
  #(questi valori mi serviranno anche per la normalizzazione inversa)

  #operazione di normalizzazione min-max sui 3 dataframe, usando i minimi e massimi del train per evitare data leakage
  train_df_[columns] = (train_df_[columns] - train_df_min) / (train_df_max - train_df_min)
  valid_df_[columns] = (valid_df_[columns] - train_df_min) / (train_df_max - train_df_min)
  test_df_[columns] = (test_df_[columns] - train_df_min) / (train_df_max - train_df_min)

  return train_df_, valid_df_, test_df_, train_df_min, train_df_max

#function to create sequences of features and target
#MODIFICATO: ora restituisce anche `idx`, l'indice originale (in `data`) della riga target di
#ciascuna sequenza. Serve a due cose:
#  1) filtrare correttamente la baseline climatology sugli stessi giorni usati per valutare
#     ogni modello (fix dello skill score richiesto dal tutor: climatology misurata su tutti
#     i giorni, modelli misurati solo sui giorni "windowed" -> confronto non comparabile);
#  2) raggruppare le previsioni per anno (2020/2021/2022) senza dover rifare il training.
#NOTA: dato che il loop è per anno e year_data non viene mai resettato sull'indice, year_index
#contiene gli indici originali del dataframe passato in input (stesso indice di *_df_clean,
#dato che normalize() copia ma non fa reset_index).
def create_sequences(data, look_back, H):
  feature_columns = [c for c in data.columns if c not in ['Date', 'year', 'Amount of irrigation']]
  target_column = 'Amount of irrigation'

  X_list = [] #feature list
  y_list = [] #target list
  idx_list = [] #NEW: indice originale (in `data`) della riga target di ciascuna sequenza

  for year in data['year'].unique():
    year_data = data[data['year'] == year]
    year_index = year_data.index.values #NEW: indici originali delle righe di questo anno

    #con values i due tipi di dati sono sotto forma di array numpy, numerato con indici 0, 1, 2...
    features = year_data[feature_columns].values  #shape (153, 8)
    target = year_data[target_column].values  #shape (153,)

    for i in range(len(year_data) - look_back - H + 1):
      window = features[i : i+look_back]  #finestra che scorre per selezionare
      target_value = target[i+look_back - 1 + H]
      target_idx = year_index[i+look_back - 1 + H]  #NEW: indice originale corrispondente al target

      X_list.append(window)
      y_list.append(target_value)
      idx_list.append(target_idx)  #NEW
    
  X = np.array(X_list)
  y = np.array(y_list)
  idx = np.array(idx_list)  #NEW

  return X, y, idx