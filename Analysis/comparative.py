import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Parámetros del experimento
strategy = "FedAvg"
strategyOpt = "FedMOpt"
data = "iid"
slow = 2
med = 0
beta = 0.85
k = 7
ex = 1
loss = [4.65, 2.35, 1.66, 1.51, 1.457, 1.4318, 1.52, 1.47] 

supernode_values = range(1, 9)
sync_values = [1, 2, 3, 4, 5, 6, 7, 9]
metrics = ["Net_receive(B)", "Net_transmit(B)", "CPU_time(s)"]
final_time = 2000
scenarios = ["S1","S2","S3","S4","S5","S6","S7","FedMOpt"]

for metric in metrics:
    # Diccionario para guardar Net_receive por supernode y sync
    metric_dict = {sn: [] for sn in supernode_values}

    # Recorrer supernodes y syncs
    for sn in supernode_values:
        for sync in sync_values:
            if sync == 9:
                csv_path = Path(
                    f"../local-execution/results/{strategyOpt}/{data}/"
                    f"slow{slow}_med{med}/"
                    f"beta{beta}/k{k}/metrics/"
                    f"supernode-{sn}/ex_{ex}.csv"
                )
                if not csv_path.exists():
                    print(f"CSV no encontrado para supernode={sn}, sync={sync}")
                    metric_dict[sn].append(0)
                    continue
                
                df = pd.read_csv(csv_path)
                df = df.sort_values("Time_stamp(s)").reset_index(drop=True)
                idx = (df["Time_stamp(s)"] - final_time).abs().idxmin()

                result = df[metric].loc[idx]
                result -= df[metric].iloc[0]
                metric_dict[sn].append(result)

            elif sync < 9:
                csv_path = Path(
                    f"../local-execution/results/{strategy}/{data}/"
                    f"sync{sync}/slow{slow}_med{med}/"
                    f"beta{beta}/k{k}/metrics/"
                    f"supernode-{sn}/ex_{ex}.csv"
                )
                if not csv_path.exists():
                    print(f"CSV no encontrado para supernode={sn}, sync={sync}")
                    metric_dict[sn].append(0)
                    continue
                
                df = pd.read_csv(csv_path)
                df = df.sort_values("Time_stamp(s)").reset_index(drop=True)
                idx = (df["Time_stamp(s)"] - final_time).abs().idxmin()

                # Tomar último Net_receive
                result = df[metric].loc[idx]
                result -= df[metric].iloc[0]
                metric_dict[sn].append(result)

    # Configuración del gráfico de barras agrupadas
    bar_width = 0.1
    x = np.arange(len(sync_values))

    plt.figure(figsize=(12,6))

    for i, sn in enumerate(supernode_values):
        plt.bar(x + i*bar_width, metric_dict[sn], width=bar_width, label=f"SN-{sn}")

    plt.xlabel("Sync")
    plt.ylabel(metric)
    plt.title("Net_receive por supernode y sync")
    plt.xticks(x + bar_width*(len(supernode_values)/2), sync_values)
    plt.legend(title="Supernode", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"prueba_{metric}.pdf", format="pdf") 
    plt.close()

    CPU_total = [sum(metric_dict[sn][sync] for sn in supernode_values)
        for sync in range(len(sync_values))]

    print(CPU_total)
    ratio = [l/c for l,c in zip(loss, CPU_total)]
    print(ratio)

    min_ratio = min(ratio)
    perc_ratio = [(r - min_ratio)/min_ratio * 100 for r in ratio]

    print(perc_ratio)

    plt.figure(figsize=(8,5))
    sc = plt.scatter(CPU_total, loss, c=ratio, cmap="viridis", s=100)
    plt.colorbar(sc, label="Loss / CPU_total")
    for i, scen in enumerate(scenarios):
        plt.text(CPU_total[i]*1.002, loss[i]*1.002, scen)
    plt.xlabel("CPU_total (s)")
    plt.ylabel("Loss")
    plt.title("Trade-off: Loss vs CPU_total (8 escenarios)")
    plt.grid(True)
    plt.savefig(f"tradeoff_{metric}.pdf", format="pdf")
