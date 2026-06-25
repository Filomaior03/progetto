import os
import random
import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from data_pipeline import *
from baselines import *
from deep_models import *

# variabili hardcoded, fisse per tutte le esecuzioni
SEED = 42
LOOK_BACK = 20
HORIZONS = [1, 3, 7, 14]
DATA_PATH = 'data/data-ready-def.xlsx'
TEST_YEARS = [2020, 2021, 2022]  #NEW: anni del test set, per la vista per-anno richiesta dal tutor


# funzione di reset dei seed (risultati e prestazioni uguali ad ogni esecuzione)
def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

#setup pipeline (LOAD, SPLIT, CLEAN)
df = load_data(DATA_PATH)
train_df, valid_df, test_df = split_data(df)

train_df_clean = clean_data(train_df)
valid_df_clean = clean_data(valid_df)
test_df_clean = clean_data(test_df)

#colonna necessaria solo per climatology_pred, le aggiungo ad una copia dei set originali
train_df_for_clim = train_df_clean.copy()
valid_df_for_clim = valid_df_clean.copy()
test_df_for_clim = test_df_clean.copy()

train_df_for_clim ['day of season'] = train_df_for_clim .groupby('year').cumcount() + 1
valid_df_for_clim['day of season'] = valid_df_for_clim.groupby('year').cumcount() + 1
test_df_for_clim['day of season'] = test_df_for_clim.groupby('year').cumcount() + 1

#normalizzazione set originali
train_df_norm, valid_df_norm, test_df_norm, train_df_min, train_df_max = normalize(train_df_clean, valid_df_clean, test_df_clean)

#massimo e minimo target (per denormalizzazione)
target_min = train_df_min['Amount of irrigation']
target_max = train_df_max['Amount of irrigation']

os.makedirs('results', exist_ok=True)

#funzione per i risultati

#metriche di climatology di riferimento adattate alla dimensione del set su cui operano i vari modelli, non più valore fisso per tutti.
def make_row(name, mae_valid, rmse_valid, r2_valid, mae_test, rmse_test, r2_test,
             clim_mae_valid, clim_rmse_valid, clim_mae_test, clim_rmse_test):
    return {'model': name, 'mae_valid': mae_valid, 'rmse_valid': rmse_valid, 'r2_valid': r2_valid,
            'mae_test': mae_test, 'rmse_test': rmse_test, 'r2_test': r2_test,
            'skill_valid': skill_score(mae_valid, clim_mae_valid),
            'skill_test': skill_score(mae_test, clim_mae_test),
            'skill_rmse_valid': skill_score(rmse_valid, clim_rmse_valid),
            'skill_rmse_test': skill_score(rmse_test, clim_rmse_test)}


#baseline indipendenti da H: le eseguo una volta sola
valid_pred_climatology = climatology_pred(train_df_for_clim, valid_df_for_clim)
test_pred_climatology = climatology_pred(train_df_for_clim, test_df_for_clim)

#NOTA: questi 3 valori (clim_valid_mae/rmse/r2, clim_test_*) restano solo per la riga
#"Climatology"

#per gli altri modelli invece, ogni riga calcola il proprio denominatore con climatology_metrics_on().
clim_valid_mae, clim_valid_rmse, clim_valid_r2 = evaluate(valid_df_clean['Amount of irrigation'], valid_pred_climatology)
clim_test_mae, clim_test_rmse, clim_test_r2 = evaluate(test_df_clean['Amount of irrigation'], test_pred_climatology)

#risultati Climatology
climatology_row = {'model': 'Climatology', 'mae_valid': clim_valid_mae, 'rmse_valid': clim_valid_rmse, 'r2_valid': clim_valid_r2,
                    'mae_test': clim_test_mae, 'rmse_test': clim_test_rmse, 'r2_test': clim_test_r2,
                    'skill_valid': None, 'skill_test': None,
                    'skill_rmse_valid': None, 'skill_rmse_test': None}  # riferimento, non ha senso uno skill su se stessa


valid_pred_seasonal, test_pred_seasonal = seasonal_pred(valid_df_clean), seasonal_pred(test_df_clean)

#filtro i NaN (rinominato da vmask/tmask per chiarezza, visto che dentro il loop ne uso
#altri due per Persistence — non voglio confondere le due maschere)
vmask_seas, tmask_seas = valid_pred_seasonal.notna(), test_pred_seasonal.notna()

#array metriche
sv = evaluate(valid_df_clean.loc[vmask_seas, 'Amount of irrigation'], valid_pred_seasonal[vmask_seas])
st = evaluate(test_df_clean.loc[tmask_seas, 'Amount of irrigation'], test_pred_seasonal[tmask_seas])

#NEW: climatology valutata sugli STESSI giorni di Seasonal naive (non su tutto il test set)
clim_v_seas = climatology_metrics_on(valid_df_clean['Amount of irrigation'], valid_pred_climatology, vmask_seas)
clim_t_seas = climatology_metrics_on(test_df_clean['Amount of irrigation'], test_pred_climatology, tmask_seas)

#risultati Seasonal Naïve
seasonal_row = make_row('Seasonal naive', sv[0], sv[1], sv[2], st[0], st[1], st[2],
                          clim_v_seas[0], clim_v_seas[1], clim_t_seas[0], clim_t_seas[1])

#NEW: skill per anno di Seasonal naive — non dipende da H, lo calcolo una sola volta e lo
#replico per ogni H nel loop sotto (stesso pattern già usato per seasonal_row)
seasonal_per_year = []
for year in TEST_YEARS:
    sel = tmask_seas & (test_df_clean['year'] == year)
    if sel.sum() == 0:
        continue
    mae_y, rmse_y, _ = evaluate(test_df_clean.loc[sel, 'Amount of irrigation'], test_pred_seasonal[sel])
    clim_mae_y, clim_rmse_y, _ = climatology_metrics_on(test_df_clean['Amount of irrigation'], test_pred_climatology, sel)
    seasonal_per_year.append({'model': 'Seasonal naive', 'year': year, 'n_days_test': int(sel.sum()),
                               'mae_test': mae_y, 'rmse_test': rmse_y,
                               'skill_test': skill_score(mae_y, clim_mae_y),
                               'skill_rmse_test': skill_score(rmse_y, clim_rmse_y)})

per_year_rows = []  #NEW: accumulatore per la tabella finale skill-per-anno (richiesta 2 del tutor)

#loop sugli orizzonti
for H in HORIZONS:
    rows = [climatology_row, seasonal_row]
    per_year_rows.extend({**r, 'H': H} for r in seasonal_per_year)  #NEW

    #unica baseline dipendente da H
    valid_pred_persistence, test_pred_persistence = persistence_pred(valid_df_clean, H), persistence_pred(test_df_clean, H)
    
    vmask, tmask = valid_pred_persistence.notna(), test_pred_persistence.notna()
    
    #risultati Persistence
    pv = evaluate(valid_df_clean.loc[vmask, 'Amount of irrigation'], valid_pred_persistence[vmask])
    pt = evaluate(test_df_clean.loc[tmask, 'Amount of irrigation'], test_pred_persistence[tmask])

    #NEW: climatology valutata sugli stessi giorni di Persistence per questo H
    clim_v_pers = climatology_metrics_on(valid_df_clean['Amount of irrigation'], valid_pred_climatology, vmask)
    clim_t_pers = climatology_metrics_on(test_df_clean['Amount of irrigation'], test_pred_climatology, tmask)
    
    #appendo i risultati a quelli delle altre baseline
    rows.append(make_row('Persistence', pv[0], pv[1], pv[2], pt[0], pt[1], pt[2],
                           clim_v_pers[0], clim_v_pers[1], clim_t_pers[0], clim_t_pers[1]))

    #NEW: skill per anno di Persistence per questo H (la maschera dipende da H -> dentro il loop)
    for year in TEST_YEARS:
        sel = tmask & (test_df_clean['year'] == year)
        if sel.sum() == 0:
            continue
        mae_y, rmse_y, _ = evaluate(test_df_clean.loc[sel, 'Amount of irrigation'], test_pred_persistence[sel])
        clim_mae_y, clim_rmse_y, _ = climatology_metrics_on(test_df_clean['Amount of irrigation'], test_pred_climatology, sel)
        per_year_rows.append({'H': H, 'model': 'Persistence', 'year': year, 'n_days_test': int(sel.sum()),
                               'mae_test': mae_y, 'rmse_test': rmse_y,
                               'skill_test': skill_score(mae_y, clim_mae_y),
                               'skill_rmse_test': skill_score(rmse_y, clim_rmse_y)})

    #divisione set in sequenze per i modelli deep
    #MODIFICATO: create_sequences ora restituisce anche l'indice originale (in *_df_norm, che
    #condivide indice con *_df_clean essendo solo una copia normalizzata) della riga target
    #di ciascuna sequenza. train_idx non serve qui, lo scarto con "_".
    train_X, train_y, _ = create_sequences(train_df_norm, LOOK_BACK, H)
    valid_X, valid_y, valid_idx = create_sequences(valid_df_norm, LOOK_BACK, H)
    test_X, test_y, test_idx = create_sequences(test_df_norm, LOOK_BACK, H)

    #NEW: climatology valutata sugli stessi giorni usati da TUTTI i modelli a sequenza
    #(LR, GB, LSTM, CNN, BiLSTM-CNN-Attention condividono lo stesso LOOK_BACK e H, quindi lo
    #stesso identico test_idx/valid_idx -> un solo calcolo, riusato per le 5 chiamate sotto)
    clim_v_seq = climatology_metrics_on(valid_df_clean['Amount of irrigation'], valid_pred_climatology, valid_idx)
    clim_t_seq = climatology_metrics_on(test_df_clean['Amount of irrigation'], test_pred_climatology, test_idx)

    #NEW: anno di ciascuna riga target test, nello stesso ordine di test_idx/test_y/test_pred
    test_years_seq = test_df_clean['year'].loc[test_idx].values

    #NEW: helper per accumulare le righe per-anno di un modello a sequenza, per non ripetere
    #4 volte lo stesso blocco (LR, GB, LSTM, CNN, BiLSTM-CNN-Attention)
    def add_per_year_seq(model_name, test_true_mm, test_pred_mm):
        for year in TEST_YEARS:
            sel = test_years_seq == year
            if sel.sum() == 0:
                continue
            mae_y, rmse_y, _ = evaluate(test_true_mm[sel], test_pred_mm[sel])
            clim_mae_y, clim_rmse_y, _ = climatology_metrics_on(
                test_df_clean['Amount of irrigation'], test_pred_climatology, test_idx[sel])
            per_year_rows.append({'H': H, 'model': model_name, 'year': year, 'n_days_test': int(sel.sum()),
                                   'mae_test': mae_y, 'rmse_test': rmse_y,
                                   'skill_test': skill_score(mae_y, clim_mae_y),
                                   'skill_rmse_test': skill_score(rmse_y, clim_rmse_y)})

    #risultati Linear Regression
    lr = fit_and_evaluate(LinearRegression(), train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
    rows.append(make_row('Linear Regression', *lr[:6], clim_v_seq[0], clim_v_seq[1], clim_t_seq[0], clim_t_seq[1]))
    add_per_year_seq('Linear Regression', lr[8], lr[9])  #NEW: test_true_mm, test_pred_mm

    #risultati Gradient Boosting
    gb = fit_and_evaluate(XGBRegressor(random_state=SEED), train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
    rows.append(make_row('Gradient Boosting', *gb[:6], clim_v_seq[0], clim_v_seq[1], clim_t_seq[0], clim_t_seq[1]))
    add_per_year_seq('Gradient Boosting', gb[8], gb[9])  #NEW

    reset_seed()
    lstm = build_lstm_model(train_X.shape[2], train_X.shape[1], lstm_units=64)

    #risultati LSTM
    #ATTENZIONE: assumo che fit_and_evaluate_deep restituisca anche (..., test_true_mm,
    #test_pred_mm) nelle posizioni [8] e [9], esattamente come fit_and_evaluate sopra. Dato
    #che in originale avevi gia' "lstm_res[:6]" (uno slice, non un unpack pieno), la funzione
    #restituisce gia' qualcosa oltre le 6 metriche: va verificato cosa c'e' in deep_models.py
    #e, se non sono gia' questi 4 array, aggiungerli con la stessa modifica fatta in
    #fit_and_evaluate() qui sopra. Mandami deep_models.py per chiudere questo punto con certezza.
    lstm_res = fit_and_evaluate_deep(lstm, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
    rows.append(make_row('LSTM', *lstm_res[:6], clim_v_seq[0], clim_v_seq[1], clim_t_seq[0], clim_t_seq[1]))
    add_per_year_seq('LSTM', lstm_res[8], lstm_res[9])  #NEW — verificare indici, vedi nota sopra

    reset_seed()
    cnn = build_cnn_model(train_X.shape[2], train_X.shape[1], filters=64)
    
    #risultati CNN
    cnn_res = fit_and_evaluate_deep(cnn, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
    rows.append(make_row('CNN', *cnn_res[:6], clim_v_seq[0], clim_v_seq[1], clim_t_seq[0], clim_t_seq[1]))
    add_per_year_seq('CNN', cnn_res[8], cnn_res[9])  #NEW — verificare indici, vedi nota sopra

    reset_seed()
    bilstm = build_bilstm_cnn_attention_model(train_X.shape[2], train_X.shape[1], lstm_units=64)
    
    #risultati BiLSTM
    bilstm_res = fit_and_evaluate_deep(bilstm, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
    rows.append(make_row('BiLSTM-CNN-Attention', *bilstm_res[:6], clim_v_seq[0], clim_v_seq[1], clim_t_seq[0], clim_t_seq[1]))
    add_per_year_seq('BiLSTM-CNN-Attention', bilstm_res[8], bilstm_res[9])  #NEW — verificare indici, vedi nota sopra

    if H == 1:
        ufficiali = {
            'LSTM': (1.686422, 2.065487, 0.166424, 1.613117, 1.908872, 0.255162),
            'CNN': (1.689530, 2.119448, 0.122300, 1.629626, 1.969049, 0.207461),
            'BiLSTM-CNN-Attention': (1.764591846719496, 2.126648124995951, 0.11632657142119429,
                                      1.6902393322131837, 1.978061936382966, 0.20018855974647864),
        }
        for row in rows:
            if row['model'] in ufficiali:
                ottenuti = (row['mae_valid'], row['rmse_valid'], row['r2_valid'],
                            row['mae_test'], row['rmse_test'], row['r2_test'])
                assert np.allclose(ottenuti, ufficiali[row['model']], atol=1e-5), \
                    f"Mismatch {row['model']} a H=1: {ottenuti} vs {ufficiali[row['model']]}"
        print("H=1: tutti i modelli deep combaciano con i numeri ufficiali del Giorno 5 — run_all.py verificato")
        print("(questo assert riguarda solo mae/rmse/r2, NON lo skill — invariati dal fix)")

    #generazione del CSV per i risultati unificati
    results_h = pd.DataFrame(rows)
    results_h.to_csv(f'results/all_models_H{H}.csv', index=False)
    
    print(f"H={H} salvato in results/all_models_H{H}.csv")
    print(results_h)
    print()

#NEW: tabella finale skill per anno (richiesta 2 del tutor) — un csv unico, tutti gli H e i modelli
per_year_df = pd.DataFrame(per_year_rows)
per_year_df.to_csv('results/master_table_per_year.csv', index=False)
print("Salvato results/master_table_per_year.csv")
print(per_year_df)