"""
build_master_table.py

Unisce i risultati per ciascun H (prodotti da run_all.py) in un'unica
results/master_table.csv — finora questo merge veniva fatto a mano.

Uso:
    python build_master_table.py

H=0 (trick test): non è prodotto da run_all.py ma da trick_test_H0.ipynb.
Se vuoi includerlo automaticamente, salva l'output di quel notebook con le
STESSE colonne di all_models_H{H}.csv (model, mae_valid, ..., skill_rmse_test,
senza la colonna H) in results/trick_test_H0.csv — vedi H0_PATH sotto.
Se il file non c'è, lo script salta H=0 con un avviso e genera comunque la
tabella per H=1/3/7/14.
"""
import os
import pandas as pd

RESULTS_DIR = 'results'
HORIZONS = [1, 3, 7, 14]
H0_PATH = os.path.join(RESULTS_DIR, 'trick_test_H0.csv')  # output di trick_test_H0.ipynb, se presente
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'master_table.csv')

#numero di righe atteso per ciascun H=1/3/7/14: Climatology, Seasonal naive, Persistence,
#Linear Regression, Gradient Boosting, LSTM, CNN, BiLSTM-CNN-Attention
EXPECTED_ROWS_PER_H = 8

frames = []

#H=0 (trick test), opzionale
if os.path.exists(H0_PATH):
    h0 = pd.read_csv(H0_PATH)
    h0.insert(0, 'H', 0)
    frames.append(h0)
    print(f"H=0 incluso da {H0_PATH} ({len(h0)} righe)")
else:
    print(f"ATTENZIONE: {H0_PATH} non trovato — master_table.csv non includerà le righe H=0.")
    print("Salva l'output di trick_test_H0.ipynb in questo percorso (stesse colonne, senza 'H') per includerlo.\n")

#H=1,3,7,14
for H in HORIZONS:
    path = os.path.join(RESULTS_DIR, f'all_models_H{H}.csv')
    df_h = pd.read_csv(path)

    #check di sanità: numero di righe atteso, per accorgersi subito se run_all.py
    #non ha completato tutti i modelli per questo H
    assert len(df_h) == EXPECTED_ROWS_PER_H, \
        f"{path} ha {len(df_h)} righe, ne aspettavo {EXPECTED_ROWS_PER_H} — controlla run_all.py per H={H}"

    df_h.insert(0, 'H', H)
    frames.append(df_h)
    print(f"H={H} incluso da {path} ({len(df_h)} righe)")

master_table = pd.concat(frames, ignore_index=True)

os.makedirs(RESULTS_DIR, exist_ok=True)
master_table.to_csv(OUTPUT_PATH, index=False)

print(f"\nSalvato {OUTPUT_PATH} — {len(master_table)} righe totali")
print(master_table)