import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_pipeline import *

#funzione di stampa per le metriche
def print_metrics(name, valid_mae, valid_rmse, valid_r2, test_mae, test_rmse, test_r2):
    print(f"--- {name} ---")
    print(f"Valid: MAE={valid_mae:.5f}  RMSE={valid_rmse:.5f}  R²={valid_r2:.4f}")
    print(f"Test:  MAE={test_mae:.5f}  RMSE={test_rmse:.5f}  R²={test_r2:.4f}")
    print()

#funzione di calcolo per le metriche
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

#funzione di addestramento e calcolo delle metriche per modelli lr/gb
def fit_and_evaluate(model, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max):
    
    #appiattisco features e target (3D -> 2D)
    train_X_flat = train_X.reshape(train_X.shape[0], -1)
    valid_X_flat = valid_X.reshape(valid_X.shape[0], -1)
    test_X_flat = test_X.reshape(test_X.shape[0], -1)

    #addestro il modello sul train
    model.fit(train_X_flat, train_y)
   
    #predizioni su valid e test (valori ancora normalizzati)
    valid_pred_norm = model.predict(valid_X_flat)
    test_pred_norm = model.predict(test_X_flat)

    #denormalizzo predizioni e target
    valid_pred_mm = valid_pred_norm * (target_max - target_min) + target_min
    test_pred_mm = test_pred_norm * (target_max - target_min) + target_min
    valid_true_mm = valid_y * (target_max - target_min) + target_min
    test_true_mm = test_y * (target_max - target_min) + target_min

    #clip: l'irrigazione non può essere negativa
    valid_pred_mm = np.clip(valid_pred_mm, a_min=0, a_max=None)
    test_pred_mm = np.clip(test_pred_mm, a_min=0, a_max=None)

    #calcolo metriche, utilizzo la funzione evaluate()
    valid_mae, valid_rmse, valid_r2 = evaluate(valid_true_mm, valid_pred_mm)
    test_mae, test_rmse, test_r2 = evaluate(test_true_mm, test_pred_mm)

    return valid_mae, valid_rmse, valid_r2, test_mae, test_rmse, test_r2

#funzione climatology
def climatology_pred(train_df, df, day_col='day of season', target_col='Amount of irrigation'):

  climatology = train_df.groupby(day_col)[target_col].mean()  #calcolo sul train la media di ogni giorno

  return df[day_col].map(climatology) #predico i valori dell'altro set (valid o test) 

#funzione persistence
def persistence_pred(df, H, target_col='Amount of irrigation', group_col='year'):
    return df.groupby(group_col)[target_col].shift(H) #per ogni anno, shifta i valori di irrigazione di H
  #Nota: prima di calcolare le metriche, necessaria una gestione dei valori NaN contenuti nel dataframe aggiornato

#funzione seasonal
def seasonal_pred(df, season_length=153, target_col='Amount of irrigation'):
   return df[target_col].shift(season_length)
  #Nota: prima di calcolare le metriche, necessaria una gestione dei valori NaN contenuti nel dataframe aggiornato