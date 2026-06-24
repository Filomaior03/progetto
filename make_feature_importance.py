import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from data_pipeline import *
from baselines import *
from deep_models import *

SEED = 42
LOOK_BACK = 20
HORIZONS = [1, 3, 7, 14]
DATA_PATH = 'data/data-ready-def.xlsx'
N_REPEATS = 10

def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ---- setup identico a run_all.py, indipendente da H ----
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

feature_columns = [c for c in train_df_norm.columns if c not in ['Date', 'year', 'Amount of irrigation']]
n_features = len(feature_columns)

# numeri ufficiali per H, da results/master_tableD.csv (già verificato ad hash)
ufficiali = {
    1: {
        'Linear Regression': (1.7398913224769466, 2.1784299914508582, 0.0727694412585663, 1.6859645157401344, 2.0310970396394006, 0.156725075711018),
        'Gradient Boosting': (1.699288154557535, 2.134419595158493, 0.1098563053291352, 1.6212543858642678, 2.03599288281752, 0.152654844194216),
        'LSTM': (1.6864219838548649, 2.0654867713133203, 0.1664237029214159, 1.613116763645489, 1.9088722425545648, 0.2551624454475638),
        'CNN': (1.6895297468516073, 2.1194476685812185, 0.1223003666945461, 1.629626081967996, 1.969048643798016, 0.2074608399261064),
        'BiLSTM-CNN-Attention': (1.764591846719496, 2.126648124995951, 0.1163265714211942, 1.6902393322131837, 1.978061936382966, 0.2001885597464786),
    },
    3: {
        'Linear Regression': (1.8985658369346103, 2.284290451514666, -0.021306045872619, 1.889736827202185, 2.204266988248729, -0.0018965369259338),
        'Gradient Boosting': (1.9706024853925708, 2.422680179619829, -0.148802644480146, 1.8917368731896465, 2.314136615459419, -0.1042628719393212),
        'LSTM': (1.87108353909976, 2.242964162636385, 0.0153136371920503, 1.829828470430271, 2.132319933890962, 0.0624396853061537),
        'CNN': (1.90036249401442, 2.290466920175303, -0.0268365097728722, 1.8603381776849504, 2.200311463151801, 0.001696014357862),
        'BiLSTM-CNN-Attention': (1.9290248143488669, 2.309195271123245, -0.0436973312609061, 1.8356182915964647, 2.1493200875328773, 0.0474304875440286),
    },
    7: {
        'Linear Regression': (1.841445810638172, 2.203334594903536, 0.0138631442295528, 1.9194292347898136, 2.231668358455898, -0.0217443437888233),
        'Gradient Boosting': (1.9777872438093265, 2.382857645398148, -0.1533801671754302, 1.9280591304867696, 2.339626422484739, -0.122990197369547),
        'LSTM': (1.8584183544056625, 2.204068006240043, 0.0132065355400931, 1.847476799476929, 2.1513768086680645, 0.050454246409177),
        'CNN': (1.8395044364356927, 2.205291467574842, 0.0121107084346658, 1.8840880643765128, 2.191291852304589, 0.0148930606699302),
        'BiLSTM-CNN-Attention': (1.909787486906862, 2.2597107100058955, -0.0372464842613431, 1.9002724503360864, 2.2388109931954943, -0.0282951607602868),
    },
    14: {
        'Linear Regression': (1.8211795352970448, 2.163622015022941, 0.0202473999272988, 1.910967576426716, 2.2433846576993592, -0.0155639698109713),
        'Gradient Boosting': (1.8856399942216568, 2.2852210119751364, -0.0929745724047559, 1.9277349233872965, 2.314817263698582, -0.0812676729860422),
        'LSTM': (1.8400382004134523, 2.181101164074532, 0.0043532963878761, 1.902599017351965, 2.225309680087051, 0.0007349270409287),
        'CNN': (1.898100587625437, 2.2626743275531624, -0.0715137288283227, 1.9214879295387537, 2.273220415624436, -0.0427564603750749),
        'BiLSTM-CNN-Attention': (1.8175336310471657, 2.1770373806207948, 0.0080599775520995, 1.8754037560853856, 2.2058959510722813, 0.0180941651799231),
    },
}

def check(name, H, ottenuti):
    assert np.allclose(ottenuti, ufficiali[H][name], atol=1e-5), \
        f"Mismatch {name} a H={H}: {ottenuti} vs {ufficiali[H][name]}"
    print(f"{name} H={H} verificato — combacia con i numeri ufficiali")


# wrapper minimale per i modelli Keras: espone solo .predict(),
# accetta input flat 2D (quello che permutation_importance vuole
# permutare) e lo reshape in 3D prima di chiamare il modello.
# Non serve .fit() né .score(): scoring='r2' più sotto chiede
# solo .predict().
class KerasFlatWrapper(RegressorMixin, BaseEstimator):
    def __init__(self, model, time_steps, n_features):
        self.model = model
        self.time_steps = time_steps
        self.n_features = n_features

    def fit(self, X, y=None):
    # non viene mai chiamato da permutation_importance (il modello è già addestrato);
    # presente solo per soddisfare la validazione dei parametri di sklearn ("estimator must implement 'fit'")
      return self

    def predict(self, X_flat):
        X_3d = np.asarray(X_flat).reshape(-1, self.time_steps, self.n_features)
        return self.model.predict(X_3d, verbose=0).flatten()

def importance_per_feature(estimator, X_flat, y):
    result = permutation_importance(estimator, X_flat, y, n_repeats=N_REPEATS,
                                     random_state=SEED, scoring='r2')
    # il flatten di (n, look_back, n_features) ordina come [t0_f0..t0_f7, t1_f0..t1_f7, ...]
    importances_grid = result.importances_mean.reshape(LOOK_BACK, n_features)
    return importances_grid.sum(axis=0)  # somma sui 20 timestep


colors = {'Linear Regression': '#1f77b4', 'Gradient Boosting': '#ff7f0e', 'LSTM': '#2ca02c',
          'CNN': '#d62728', 'BiLSTM-CNN-Attention': '#9467bd'}
model_order = ['Linear Regression', 'Gradient Boosting', 'LSTM', 'CNN', 'BiLSTM-CNN-Attention']

for H in HORIZONS:
    print(f"=== H={H} ===")

    train_X, train_y = create_sequences(train_df_norm, LOOK_BACK, H)
    valid_X, valid_y = create_sequences(valid_df_norm, LOOK_BACK, H)
    test_X, test_y = create_sequences(test_df_norm, LOOK_BACK, H)

    train_X_flat = train_X.reshape(train_X.shape[0], -1)
    test_X_flat = test_X.reshape(test_X.shape[0], -1)

    all_importances = {}

    # ---------------------------------------------------------
    # Linear Regression
    # ---------------------------------------------------------
    lr = LinearRegression()
    lr_res = fit_and_evaluate(lr, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
    check('Linear Regression', H, lr_res)
    print("Linear Regression verificato — procedo con permutation importance")
    all_importances['Linear Regression'] = importance_per_feature(lr, test_X_flat, test_y)

    # ---------------------------------------------------------
    # Gradient Boosting
    # ---------------------------------------------------------
    gb = XGBRegressor(random_state=SEED)
    gb_res = fit_and_evaluate(gb, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
    check('Gradient Boosting', H, gb_res)
    print("Gradient Boosting verificato — procedo con permutation importance")
    all_importances['Gradient Boosting'] = importance_per_feature(gb, test_X_flat, test_y)

    # ---------------------------------------------------------
    # LSTM
    # ---------------------------------------------------------
    reset_seed()
    lstm = build_lstm_model(train_X.shape[2], train_X.shape[1], lstm_units=64)
    lstm_res = fit_and_evaluate_deep(lstm, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
    check('LSTM', H, lstm_res[:6])
    print("LSTM verificato — procedo con permutation importance")
    lstm_wrapped = KerasFlatWrapper(lstm, LOOK_BACK, n_features)
    all_importances['LSTM'] = importance_per_feature(lstm_wrapped, test_X_flat, test_y)

    # ---------------------------------------------------------
    # CNN
    # ---------------------------------------------------------
    reset_seed()
    cnn = build_cnn_model(train_X.shape[2], train_X.shape[1], filters=64)
    cnn_res = fit_and_evaluate_deep(cnn, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
    check('CNN', H, cnn_res[:6])
    print("CNN verificato — procedo con permutation importance")
    cnn_wrapped = KerasFlatWrapper(cnn, LOOK_BACK, n_features)
    all_importances['CNN'] = importance_per_feature(cnn_wrapped, test_X_flat, test_y)

    # ---------------------------------------------------------
    # BiLSTM-CNN-Attention
    # ---------------------------------------------------------
    reset_seed()
    bilstm = build_bilstm_cnn_attention_model(train_X.shape[2], train_X.shape[1], lstm_units=64)
    bilstm_res = fit_and_evaluate_deep(bilstm, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max)
    check('BiLSTM-CNN-Attention', H, bilstm_res[:6])
    print("BiLSTM-CNN-Attention verificato — procedo con permutation importance")
    bilstm_wrapped = KerasFlatWrapper(bilstm, LOOK_BACK, n_features)
    all_importances['BiLSTM-CNN-Attention'] = importance_per_feature(bilstm_wrapped, test_X_flat, test_y)

    # =========================================================
    # Salvataggio risultati in CSV (formato long: feature, modello, importanza)
    # =========================================================
    rows = []
    for model_name, importances in all_importances.items():
        for feat, imp in zip(feature_columns, importances):
            rows.append({'model': model_name, 'feature': feat, 'importance': imp})

    importance_df = pd.DataFrame(rows)
    importance_df.to_csv(f'results/feature_importance_all_models_H{H}.csv', index=False)
    print(f"Salvato results/feature_importance_all_models_H{H}.csv")

    # =========================================================
    # Fig.4 — Feature importance, confronto tra tutti i modelli, H
    # =========================================================
    x = np.arange(n_features)
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model_name in enumerate(model_order):
        offset = (i - (len(model_order) - 1) / 2) * width
        ax.bar(x + offset, all_importances[model_name], width, label=model_name, color=colors[model_name])

    ax.set_xticks(x)
    ax.set_xticklabels(feature_columns, rotation=30, ha='right')
    ax.set_ylabel('Permutation importance (calo medio di R² test)')
    ax.set_title(f"Fig. 4 — Feature importance, confronto tra tutti i modelli, H={H}\n(somma sui 20 timestep della finestra)")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend()
    ax.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(f'figures/fig4_feature_importance_all_models_H{H}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Fig.4 H={H} (confronto tutti i modelli) salvata")
    print()