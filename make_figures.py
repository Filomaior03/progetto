import pandas as pd
import matplotlib.pyplot as plt
import os

def load_clean_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df

h0 = load_clean_csv('results/trick_test_H0.csv')
h0['H'] = 0

horizons_files = {1: 'results/all_models_H1.csv', 3: 'results/all_models_H3.csv',
                   7: 'results/all_models_H7.csv', 14: 'results/all_models_H14.csv'}
frames = [h0]
for H, path in horizons_files.items():
    df = load_clean_csv(path)
    df['H'] = H
    frames.append(df)

all_data = pd.concat(frames, ignore_index=True)

h_values = [0, 1, 3, 7, 14]
h_to_pos = {h: i for i, h in enumerate(h_values)}
all_data['x'] = all_data['H'].map(h_to_pos)

feature_models = ['Linear Regression', 'Gradient Boosting', 'LSTM', 'CNN', 'BiLSTM-CNN-Attention']
baseline_models = ['Climatology', 'Persistence', 'Seasonal naive']
colors = {'Linear Regression': '#1f77b4', 'Gradient Boosting': '#ff7f0e', 'LSTM': '#2ca02c',
          'CNN': '#d62728', 'BiLSTM-CNN-Attention': '#9467bd'}

fig, ax = plt.subplots(figsize=(10, 6))
ax.axvspan(-0.3, 0.5, color='red', alpha=0.06)
ax.text(0, 1.0, 'H=0\n(leakage)', ha='center', va='bottom', fontsize=9, color='#A32D2D')

for model in baseline_models:
    sub = all_data[all_data['model'] == model].sort_values('x')
    ax.plot(sub['x'], sub['r2_test'], linestyle='--', color='gray', alpha=0.6, marker='o', markersize=4)
ax.plot([], [], linestyle='--', color='gray', alpha=0.6, label='Baseline naive')

for model in feature_models:
    sub = all_data[all_data['model'] == model].sort_values('x')
    ax.plot(sub['x'], sub['r2_test'], marker='o', color=colors[model], label=model, linewidth=2)

ax.set_xticks(range(len(h_values)))
ax.set_xticklabels([str(h) for h in h_values])
ax.set_xlabel('Orizzonte di previsione H (giorni)')
ax.set_ylabel('R² (test set)')
ax.set_title("Degrado delle prestazioni al crescere dell'orizzonte: H=0 (leakage) → H=14")
ax.axhline(0, color='black', linewidth=0.8)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.2)
ax.set_ylim(-0.95, 1.1)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/horizons_fig.png', dpi=300, bbox_inches='tight')
plt.show()