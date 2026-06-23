# import os
# import random
# import pandas as pd
# import tensorflow as tf
# import numpy as np

# from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor

# from data_pipeline import *
# from baselines import *
# from deep_models import *

# # variabili hardcoded, fisse per tutte le esecuzioni
# SEED = 42
# LOOK_BACK = 20
# HORIZONS = [1, 3, 7, 14]
# DATA_PATH = 'data/data-ready-def.xlsx'


# # funzione di reset dei seed (risultati e prestazioni uguali ad ogni esecuzione)
# def reset_seed():
#     random.seed(SEED)
#     np.random.seed(SEED)
#     tf.random.set_seed(SEED)

# #setup pipeline (LOAD, SPLIT, CLEAN)
# df = load_data(DATA_PATH)
# train_df, valid_df, test_df = split_data(df)

# train_df_clean = clean_data(train_df)
# valid_df_clean = clean_data(valid_df)
# test_df_clean = clean_data(test_df)

# #colonna necessaria solo per climatology_pred, le aggiungo ad una copia dei set originali
# train_df_for_clim = train_df_clean.copy()
# valid_df_for_clim = valid_df_clean.copy()
# test_df_for_clim = test_df_clean.copy()

# train_df_for_clim ['day of season'] = train_df_for_clim .groupby('year').cumcount() + 1
# valid_df_for_clim['day of season'] = valid_df_for_clim.groupby('year').cumcount() + 1
# test_df_for_clim['day of season'] = test_df_for_clim.groupby('year').cumcount() + 1

# #normalizzazione set originali
# train_df_norm, valid_df_norm, test_df_norm, train_df_min, train_df_max = normalize(train_df_clean, valid_df_clean, test_df_clean)

# #massimo e minimo target (per denormalizzazione)
# target_min = train_df_min['Amount of irrigation']
# target_max = train_df_max['Amount of irrigation']

# os.makedirs('results', exist_ok=True)

# #funzione per i risultati
# def make_row(name, mae_valid, rmse_valid, r2_valid, mae_test, rmse_test, r2_test):
#     return {'model': name, 'mae_valid': mae_valid, 'rmse_valid': rmse_valid, 'r2_valid': r2_valid,
#             'mae_test': mae_test, 'rmse_test': rmse_test, 'r2_test': r2_test,
#             'skill_valid': skill_score(mae_valid, clim_valid_mae),
#             'skill_test': skill_score(mae_test, clim_test_mae)}


# #baseline indipendenti da H: le eseguo una volta sola
# valid_pred_climatology = climatology_pred(train_df_for_clim, valid_df_for_clim)
# test_pred_climatology = climatology_pred(train_df_for_clim, test_df_for_clim)

# clim_valid_mae, clim_valid_rmse, clim_valid_r2 = evaluate(valid_df_clean['Amount of irrigation'], valid_pred_climatology)
# clim_test_mae, clim_test_rmse, clim_test_r2 = evaluate(test_df_clean['Amount of irrigation'], test_pred_climatology)

# #risultati Climatology
# climatology_row = {'model': 'Climatology', 'mae_valid': clim_valid_mae, 'rmse_valid': clim_valid_rmse, 'r2_valid': clim_valid_r2,
#                     'mae_test': clim_test_mae, 'rmse_test': clim_test_rmse, 'r2_test': clim_test_r2,
#                     'skill_valid': None, 'skill_test': None}  # riferimento, non ha senso uno skill su se stessa


# valid_pred_seasonal, test_pred_seasonal = seasonal_pred(valid_df_clean), seasonal_pred(test_df_clean)

# #filtro i NaN
# vmask, tmask = valid_pred_seasonal.notna(), test_pred_seasonal.notna()

# #array metriche
# sv = evaluate(valid_df_clean.loc[vmask, 'Amount of irrigation'], valid_pred_seasonal[vmask])
# st = evaluate(test_df_clean.loc[tmask, 'Amount of irrigation'], test_pred_seasonal[tmask])

# #risultati Seasonal Naïve
# seasonal_row = make_row('Seasonal naive', sv[0], sv[1], sv[2], st[0], st[1], st[2])

# #loop sugli orizzonti
# for H in HORIZONS:
#     rows = [climatology_row, seasonal_row]

#     #unica baseline dipendente da H
#     valid_pred_persistence, test_pred_persistence = persistence_pred(valid_df_clean, H), persistence_pred(test_df_clean, H)
    
#     vmask, tmask = valid_pred_persistence.notna(), test_pred_persistence.notna()
    
#     #risultati Persistence
#     pv = evaluate(valid_df_clean.loc[vmask, 'Amount of irrigation'], valid_pred_persistence[vmask])
#     pt = evaluate(test_df_clean.loc[tmask, 'Amount of irrigation'], test_pred_persistence[tmask])
    
#     #appendo i risultati a quelli delle altre baseline
#     rows.append(make_row('Persistence', pv[0], pv[1], pv[2], pt[0], pt[1], pt[2]))

#     #divisione set in sequenze per i modelli deep
#     train_X, train_y = create_sequences(train_df_norm, LOOK_BACK, H)
#     valid_X, valid_y = create_sequences(valid_df_norm, LOOK_BACK, H)
#     test_X, test_y = create_sequences(test_df_norm, LOOK_BACK, H)

#     #risultati Linear Regression
#     lr = fit_and_evaluate(LinearRegression(), train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
#     rows.append(make_row('Linear Regression', *lr))

#     #risultati Gradient Boosting
#     gb = fit_and_evaluate(XGBRegressor(random_state=SEED), train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
#     rows.append(make_row('Gradient Boosting', *gb))

#     reset_seed()
#     lstm = build_lstm_model(train_X.shape[2], train_X.shape[1], lstm_units=64)

#     #risultati LSTM
#     lstm_res = fit_and_evaluate_deep(lstm, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
#     rows.append(make_row('LSTM', *lstm_res[:6]))

#     reset_seed()
#     cnn = build_cnn_model(train_X.shape[2], train_X.shape[1], filters=64)
    
#     #risultati CNN
#     cnn_res = fit_and_evaluate_deep(cnn, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
#     rows.append(make_row('CNN', *cnn_res[:6]))

#     reset_seed()
#     bilstm = build_bilstm_cnn_attention_model(train_X.shape[2], train_X.shape[1], lstm_units=64)
    
#     #risultati BiLSTM
#     bilstm_res = fit_and_evaluate_deep(bilstm, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
#     rows.append(make_row('BiLSTM-CNN-Attention', *bilstm_res[:6]))

#     if H == 1:
#         ufficiali = {
#             'LSTM': (1.686422, 2.065487, 0.166424, 1.613117, 1.908872, 0.255162),
#             'CNN': (1.689530, 2.119448, 0.122300, 1.629626, 1.969049, 0.207461),
#             'BiLSTM-CNN-Attention': (1.764591846719496, 2.126648124995951, 0.11632657142119429,
#                                       1.6902393322131837, 1.978061936382966, 0.20018855974647864),
#         }
#         for row in rows:
#             if row['model'] in ufficiali:
#                 ottenuti = (row['mae_valid'], row['rmse_valid'], row['r2_valid'],
#                             row['mae_test'], row['rmse_test'], row['r2_test'])
#                 assert np.allclose(ottenuti, ufficiali[row['model']], atol=1e-5), \
#                     f"Mismatch {row['model']} a H=1: {ottenuti} vs {ufficiali[row['model']]}"
#         print("H=1: tutti i modelli deep combaciano con i numeri ufficiali del Giorno 5 — run_all.py verificato")

#     #generazione del CSV per i risultati unificati
#     results_h = pd.DataFrame(rows)
#     results_h.to_csv(f'results/all_models_H{H}.csv', index=False)
    
#     print(f"H={H} salvato in results/all_models_H{H}.csv")
#     print(results_h)
#     print()


import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from data_pipeline import *
from baselines import *
from deep_models import *

SEED = 42
LOOK_BACK = 20
H = 1
DATA_PATH = 'data/data-ready-def.xlsx'

def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ---- setup identico a run_all.py, solo H=1 ----
df = load_data(DATA_PATH)
train_df, valid_df, test_df = split_data(df)
train_df_clean = clean_data(train_df)
valid_df_clean = clean_data(valid_df)
test_df_clean = clean_data(test_df)

train_df_norm, valid_df_norm, test_df_norm, train_df_min, train_df_max = normalize(
    train_df_clean, valid_df_clean, test_df_clean
)
target_min = train_df_min['Amount of irrigation']
target_max = train_df_max['Amount of irrigation']

train_X, train_y = create_sequences(train_df_norm, LOOK_BACK, H)
valid_X, valid_y = create_sequences(valid_df_norm, LOOK_BACK, H)
test_X, test_y = create_sequences(test_df_norm, LOOK_BACK, H)

ufficiali = {
    'LSTM': (1.686422, 2.065487, 0.166424, 1.613117, 1.908872, 0.255162),
    'BiLSTM-CNN-Attention': (1.764591846719496, 2.126648124995951, 0.11632657142119429,
                              1.6902393322131837, 1.978061936382966, 0.20018855974647864),
}

# =========================================================
# FIG 2 — Predicted vs actual, LSTM (miglior modello), H=1
# =========================================================
reset_seed()
lstm = build_lstm_model(train_X.shape[2], train_X.shape[1], lstm_units=64)
r = fit_and_evaluate_deep(lstm, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
lstm_valid_mae, lstm_valid_rmse, lstm_valid_r2, lstm_test_mae, lstm_test_rmse, lstm_test_r2, lstm_history = r

ottenuti = (lstm_valid_mae, lstm_valid_rmse, lstm_valid_r2, lstm_test_mae, lstm_test_rmse, lstm_test_r2)
assert np.allclose(ottenuti, ufficiali['LSTM'], atol=1e-5), f"Mismatch LSTM: {ottenuti}"
print("LSTM H=1 verificato — procedo con le predizioni per Fig.2")

test_pred_norm = lstm.predict(test_X).flatten()
test_pred_mm = np.clip(test_pred_norm * (target_max - target_min) + target_min, a_min=0, a_max=None)
test_true_mm = test_y * (target_max - target_min) + target_min

np.save('results/lstm_H1_test_preds.npy', test_pred_mm)
np.save('results/lstm_H1_test_true.npy', test_true_mm)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
ax.scatter(test_true_mm, test_pred_mm, alpha=0.4, s=15, color='#2ca02c')
lims = [0, max(test_true_mm.max(), test_pred_mm.max()) * 1.05]
ax.plot(lims, lims, 'k--', linewidth=1, label='predizione perfetta')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Irrigazione reale (mm)')
ax.set_ylabel('Irrigazione predetta (mm)')
ax.set_title(f'Scatter (R²={lstm_test_r2:.3f})')
ax.legend()
ax.grid(alpha=0.2)

ax = axes[1]
ax.plot(test_true_mm, label='Reale', color='black', linewidth=1.2)
ax.plot(test_pred_mm, label='Predetto (LSTM)', color='#2ca02c', linewidth=1.2, alpha=0.85)
ax.set_xlabel('Giorno nel test set (sequenziale, 2020-2022)')
ax.set_ylabel('Irrigazione (mm)')
ax.set_title('Andamento temporale: reale vs predetto')
ax.legend()
ax.grid(alpha=0.2)

plt.suptitle('Fig. 2 — Predicted vs actual, LSTM (miglior modello), H=1, test set 2020-2022')
plt.tight_layout()
plt.savefig('figures/fig2_predicted_vs_actual_H1.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Fig.2 salvata")

# =========================================================
# FIG 3 — Training loss curve, BiLSTM-CNN-Attention, H=1
# =========================================================
reset_seed()
bilstm = build_bilstm_cnn_attention_model(train_X.shape[2], train_X.shape[1], lstm_units=64)
r = fit_and_evaluate_deep(bilstm, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
bilstm_valid_mae, bilstm_valid_rmse, bilstm_valid_r2, bilstm_test_mae, bilstm_test_rmse, bilstm_test_r2, bilstm_history = r

ottenuti = (bilstm_valid_mae, bilstm_valid_rmse, bilstm_valid_r2, bilstm_test_mae, bilstm_test_rmse, bilstm_test_r2)
assert np.allclose(ottenuti, ufficiali['BiLSTM-CNN-Attention'], atol=1e-9), f"Mismatch BiLSTM-CNN-Attention: {ottenuti}"
print("BiLSTM-CNN-Attention H=1 verificato — procedo con la loss curve per Fig.3")

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(bilstm_history.history['loss'], label='Training loss', color='#9467bd')
ax.plot(bilstm_history.history['val_loss'], label='Validation loss', color='#d62728')
ax.set_xlabel('Epoca')
ax.set_ylabel('MSE (su dati normalizzati)')
ax.set_title('Fig. 3 — Training loss curve, BiLSTM-CNN-Attention, H=1')
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('figures/fig3_training_loss_bilstm.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"Fig.3 salvata (early stop a epoca {len(bilstm_history.history['loss'])})")

# =========================================================
# FIG 4 — Feature importance, Gradient Boosting, H=1
# =========================================================
train_X_flat = train_X.reshape(train_X.shape[0], -1)
test_X_flat = test_X.reshape(test_X.shape[0], -1)

gb = XGBRegressor(random_state=SEED)
gb.fit(train_X_flat, train_y)

test_pred_norm_gb = gb.predict(test_X_flat)
test_pred_mm_gb = np.clip(test_pred_norm_gb * (target_max - target_min) + target_min, a_min=0, a_max=None)
gb_test_mae, gb_test_rmse, gb_test_r2 = evaluate(test_true_mm, test_pred_mm_gb)
print(f"GB H=1 test MAE={gb_test_mae:.6f} (ufficiale: 1.621254)")
assert np.isclose(gb_test_mae, 1.621254, atol=1e-4), f"Mismatch Gradient Boosting H=1: {gb_test_mae}"
print("Gradient Boosting H=1 verificato — procedo con permutation importance per Fig.4")

result = permutation_importance(gb, test_X_flat, test_y, n_repeats=10, random_state=SEED)

feature_columns = [c for c in train_df_norm.columns if c not in ['Date', 'year', 'Amount of irrigation']]
n_features = len(feature_columns)

# il flatten di (n, look_back, n_features) ordina come [t0_f0..t0_f7, t1_f0..t1_f7, ...]
importances_grid = result.importances_mean.reshape(LOOK_BACK, n_features)
feature_importance = importances_grid.sum(axis=0)

order = np.argsort(feature_importance)[::-1]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.bar(range(n_features), feature_importance[order], color='#ff7f0e')
ax.set_xticks(range(n_features))
ax.set_xticklabels([feature_columns[i] for i in order], rotation=30, ha='right')
ax.set_ylabel('Permutation importance (calo medio di R² test)')
ax.set_title('Fig. 4 — Feature importance, Gradient Boosting, H=1\n(somma sui 20 timestep della finestra)')
ax.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('figures/fig4_feature_importance_H1.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Fig.4 salvata")