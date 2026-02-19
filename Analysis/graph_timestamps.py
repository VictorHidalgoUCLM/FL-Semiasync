import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re

strategy = "FedAvg"

# -------------------------------------------------
# Función para extraer número de cliente
# -------------------------------------------------
def cliente_numero(cliente):
    match = re.search(r'\d+', cliente)
    return int(match.group()) if match else 0


# -------------------------------------------------
# Loop: una figura (9x7) por PDF
# -------------------------------------------------
for k in [2]:

    if k == 3:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig, ax = plt.subplots(figsize=(9, 7))

    path = f"../local-execution/results/{strategy}/iid/sync{k}/slow1_med0/beta1.0/gamma0.9/logs/timestamp_1.csv"
    df = pd.read_csv(path)

    intervalos = {}
    server_ev = df["server_ev"].values

    # -------------------------------------------------
    # Construcción de intervalos
    # -------------------------------------------------
    for col in df.columns:
        if col.endswith("_fit") and col != "server_fit":
            cliente = col.replace("_fit", "")
            fits = df[col].values

            runs = []
            last_fit = None
            start = 0.0

            for i, fit_time in enumerate(fits):
                if pd.isna(fit_time) or fit_time == "{}":
                    continue

                if last_fit is None or fit_time != last_fit:
                    end = float(fit_time)
                    if end > start:
                        runs.append((start, end))

                    last_fit = fit_time
                    start = server_ev[i]

            if runs:
                intervalos[cliente] = runs

    # -------------------------------------------------
    # Orden y colores
    # -------------------------------------------------
    clientes_ordenados = sorted(intervalos.keys(), key=cliente_numero)[::-1]
    colores = list(mcolors.TABLEAU_COLORS.values())
    cliente_color = {
        c: colores[i % len(colores)] for i, c in enumerate(clientes_ordenados)
    }

    # -------------------------------------------------
    # w^0_G
    # -------------------------------------------------
    ax.text(
        0,
        len(clientes_ordenados) - 0.3,
        r"$w^0_G$",
        ha="center",
        va="bottom",
        fontsize=24,
        clip_on=False
    )

    # -------------------------------------------------
    # Flechas
    # -------------------------------------------------
    for i, cliente in enumerate(clientes_ordenados):
        for a, b in intervalos[cliente]:
            ax.arrow(
                a, i,
                b - a - 1.0, 0,
                width=0.3,
                length_includes_head=True,
                head_width=0.6,
                head_length=20,
                color=cliente_color[cliente],
                linewidth=2,
                zorder=1
            )

        if i == 2:
            print(intervalos[cliente])
            x_lim = intervalos[cliente][1][1]

    # -------------------------------------------------
    # Líneas verticales y w^i_G
    # -------------------------------------------------
    for i, fit in enumerate(df["server_fit"]):
        if fit < x_lim + 1.0:
            ax.vlines(
                fit,
                -0.5,
                len(clientes_ordenados) - 0.5,
                color="black",
                linestyle=(5, (10, 3)),
                linewidth=2,
                zorder=10
            )
            ax.text(
                fit,
                len(clientes_ordenados) - 0.3,
                rf"$w^{{{i+1}}}_G$",
                ha="center",
                va="bottom",
                fontsize=24,
                clip_on=False
            )
        else:
            break

    # Última línea
    ax.vlines(
        x_lim - 0.4,
        -0.5,
        len(clientes_ordenados) - 0.5,
        color="black",
        linestyle=(5, (10, 3)),
        linewidth=2,
        zorder=10
    )

    # -------------------------------------------------
    # Estética
    # -------------------------------------------------
    ax.set_xlim(0, x_lim)
    ax.set_ylim(-0.5)
    ax.set_xticks([])
    
    if k == 3:
        ax.set_yticks(range(len(clientes_ordenados)))
        ax.set_yticklabels(
            ["Client 1", "Client 2", "Client 3"],
            fontsize=26
        )

    else:
        ax.set_yticks([])

    ax.set_xlabel("Time", fontsize=26, loc="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"timestamp_{strategy}_sync{k}.pdf", format="pdf")
    plt.close(fig)
