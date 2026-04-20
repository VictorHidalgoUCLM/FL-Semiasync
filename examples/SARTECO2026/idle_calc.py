import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import numpy as np

out_dir = Path('output')
out_dir.mkdir(exist_ok=True)

COLORS_BY_METHOD = {
    'sync5': '#1f77b4',
    'sync6': '#ff7f0e',
    'sync7': '#2ca02c',
    'FedMDyn': '#d62728'
}

def read_and_compute(file_path):
    df = pd.read_csv(file_path)

    server_fit = pd.to_numeric(df['server_fit'], errors='coerce')
    server_ev = pd.to_numeric(df['server_ev'], errors='coerce')
    ev_times = server_ev - server_fit

    idle = pd.DataFrame(index=df.index)
    exec_df = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col.endswith('_fit') and col != 'server_fit':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mask = df[col].diff().fillna(1) != 0
            name = col.replace('_fit', '')
            idle[name] = (server_fit - df[col]).where(mask).clip(lower=0)
            exec_df[name] = ev_times.where(mask)

    last_value = server_ev.iloc[-1]
    suma_idle = idle.sum()
    suma_ev = exec_df.sum()
    exec_times = last_value - suma_ev - suma_idle

    df_plot = pd.DataFrame({
        'evaluation': suma_ev,
        'idle': suma_idle,
        'execution': exec_times
    }).sort_index()

    df_plot['total'] = df_plot['evaluation'] + df_plot['idle'] + df_plot['execution']
    df_plot['ev_pct'] = (df_plot['evaluation'] / df_plot['total']) * 100
    df_plot['idle_pct'] = (df_plot['idle'] / df_plot['total']) * 100
    df_plot['exec_pct'] = (df_plot['execution'] / df_plot['total']) * 100

    return df_plot

def average_timestamp_runs(folder_path):
    files = sorted(glob.glob(os.path.join(folder_path, 'timestamp_*.csv')))
    if not files:
        return None, None

    results = [read_and_compute(f) for f in files]
    mean_df = pd.concat(results).groupby(level=0).mean(numeric_only=True)
    std_df = pd.concat(results).groupby(level=0).std(numeric_only=True)
    return mean_df, std_df

experiments = {
    'sync5': '/home/usuario/Escritorio/FL-Semiasync/local-execution/results/FedAvg/iid/sync5/slow2/beta0.7/gamma12.0/logs',
    'sync6': '/home/usuario/Escritorio/FL-Semiasync/local-execution/results/FedAvg/iid/sync6/slow2/beta0.7/gamma12.0/logs',
    'sync7': '/home/usuario/Escritorio/FL-Semiasync/local-execution/results/FedAvg/iid/sync7/slow2/beta0.7/gamma12.0/logs',
    'FedMDyn': '/home/usuario/Escritorio/FL-Semiasync/local-execution/results/FedMOpt/iid/slow2/beta0.7/gamma12.0/logs'
}

data_summary = {}
for exp_name, folder_path in experiments.items():
    if os.path.exists(folder_path):
        mean_df, std_df = average_timestamp_runs(folder_path)
        if mean_df is not None:
            data_summary[exp_name] = {
                'evaluation': (mean_df['evaluation'].mean(), std_df['evaluation'].mean()),
                'idle': (mean_df['idle'].mean(), std_df['idle'].mean()),
                'execution': (mean_df['execution'].mean(), std_df['execution'].mean())
            }

if data_summary:
    categories = ['Evaluation', 'Idle', 'Execution']
    exp_names = list(data_summary.keys())
    n_exps = len(exp_names)

    bar_width = 0.2
    x_pos = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, exp_name in enumerate(exp_names):
        means = [data_summary[exp_name][cat.lower()][0] for cat in categories]
        stds = [data_summary[exp_name][cat.lower()][1] for cat in categories]

        if exp_name == "FedMDyn":
            color = COLORS_BY_METHOD["FedMDyn"]
            label = "FedMDyn"
        else:
            color = COLORS_BY_METHOD[exp_name]
            label = f"M = {exp_name.replace('sync', '')}"

        ax.bar(
            x_pos + i * bar_width,
            means,
            bar_width,
            label=label,
            alpha=0.8,
            color=color
        )

    ax.set_xticks(x_pos + bar_width * (n_exps - 1) / 2)
    ax.set_xticklabels(categories)
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.set_xlabel('Componente temporal', fontsize=18)
    ax.set_ylabel('Tiempo medio (s)', fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'time_decomposition_summary.pdf', bbox_inches='tight')
    plt.close()
