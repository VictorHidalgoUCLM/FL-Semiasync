import pandas as pd
from pathlib import Path
import numpy as np

# Parámetros del experimento (ejemplo)
strategy = "FedAvg"
strategyComp = "FedMOpt"
data = "iid"
sync = 5
slow = 0
med = 0
beta = 0.7
k = 8
supernode = 1
ex = 1

# Construcción de la ruta
csv_path_base = Path(
    f"../local-execution/results/{strategy}/{data}/"
    f"sync{sync}/slow{slow}_med{med}/"
    f"beta{beta}/k{k}/metrics/"
    f"supernode-{supernode}/ex_{ex}.csv"
)

csv_path_comp = Path(
    f"../local-execution/results/{strategyComp}/{data}/"
    f"/slow{slow}_med{med}/"
    f"beta{beta}/k{k}/metrics/"
    f"supernode-{supernode}/ex_{ex}.csv"
)

# Cargar CSV
df_base = pd.read_csv(csv_path_base)
df_comp = pd.read_csv(csv_path_comp)

df_base = df_base.sort_values("Time_stamp(s)").reset_index(drop=True)
df_comp = df_comp.sort_values("Time_stamp(s)").reset_index(drop=True)

print(df_base.head())
print(df_comp.head())

# Delta de tiempo entre muestras
df_base["dt"] = df_base["Time_stamp(s)"].diff()

df_base["net_tx_Bps"] = df_base["Net_transmit(B)"].diff() / df_base["dt"]
df_base["net_rx_Bps"] = df_base["Net_receive(B)"].diff() / df_base["dt"]

df_base["cpu_time_delta"] = df_base["CPU_time(s)"].diff()
df_base["cpu_throttled_delta"] = df_base["CPU_throttled_time(s)"].diff()
df_base["num_cpus"] = df_base["cpu_client(quota)"] / 100000

df_base["cpu_usage"] = df_base["cpu_time_delta"] / (df_base["dt"] * df_base["num_cpus"]) * 100

df_base["cpu_demand"] = (
    df_base["cpu_time_delta"] + df_base["cpu_throttled_delta"]
) / (df_base["dt"] * df_base["num_cpus"])

df_base["cpu_pressure"] = (
    df_base["cpu_throttled_delta"]
    / (df_base["cpu_time_delta"] + df_base["cpu_throttled_delta"])
)
df_base["cpu_pressure_pct"] = 100 * df_base["cpu_pressure"]

df_base["ram_usage_pct"] = 100 * df_base["RAM_usage(B)"] / df_base["memory_client(B)"]
df_base["swap_usage_pct"] = 100 * df_base["Swap_usage(B)"] / df_base["swap_client(B)"]

df_base["ram_usage_pct"] = 100 * df_base["RAM_usage(B)"] / df_base["memory_client(B)"]
df_base["swap_usage_pct"] = 100 * df_base["Swap_usage(B)"] / df_base["swap_client(B)"]

df_clean = df_base.drop(columns=[
    "cpu_time_delta",
    "cpu_throttled_delta",
    "num_cpus"
])

df_clean = df_clean.dropna(subset=["dt"])

#print(df_clean)