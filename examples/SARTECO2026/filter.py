import matplotlib.pyplot as plt
import re
from pathlib import Path

colors = ['#5e548e', '#4a7c59', '#b56576']

def extract_M_values(file_path):
    vals = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.search(r"M\s*=\s*(\d+)", line)
                if m:
                    vals.append(int(m.group(1)))
    except FileNotFoundError:
        pass
    return vals

def process_and_plot_selected(root_dir):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        print(f"❌ Directorio no existe: {root_dir}")
        return

    cleaned_files = list(root_dir.glob("*_cleaned.out"))
    numbers = [
        int(re.match(r"(\d+)_cleaned\.out", f.name).group(1))
        for f in cleaned_files
        if re.match(r"(\d+)_cleaned\.out", f.name)
    ]

    if not numbers:
        print("❌ No se encontraron *_cleaned.out")
        return

    max_n = max(numbers)
    selected_files = list(range(0, max_n + 1, 3))[:3]

    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)

    titles = ["slow0", "slow1", "slow2"]

    for idx, (n, title, color) in enumerate(zip(selected_files, titles, colors)):
        file_path = root_dir / f"{n}_cleaned.out"
        vals = extract_M_values(file_path)

        fig, ax = plt.subplots(figsize=(6, 4))

        if vals:
            ax.plot(vals, marker='o', markersize=3, linewidth=1.5, color=color)
            #ax.set_title(title, fontsize=13)
            ax.set_ylim(1 - 0.5, 7 + 0.5)
            ax.set_xlim(0-0.2, len(vals) - 1+0.2)
            ax.set_yticks(range(1, 8))
            ax.grid(True, alpha=0.3)

            if idx == 0:
                ax.set_ylabel(r"Valores $M_{Dyn}$", fontsize=18)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)

            ax.set_xlabel("Ronda", fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
        else:
            ax.text(0.5, 0.5, f"{title}: sin datos", ha='center', va='center')
            ax.set_axis_off()

        plt.tight_layout()
        plt.savefig(out_dir / f"M_evolution_{title}.pdf", bbox_inches='tight')
        plt.close()

# ==============================
# EJECUTAR
# ==============================
root_directory = "/home/usuario/Escritorio/FL-Semiasync/local-execution/results/FedMOpt/1062781"
process_and_plot_selected(root_directory)