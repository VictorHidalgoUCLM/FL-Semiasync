import random, yaml, os

N_CLIENTES = 8          # número de clientes
FASES = 20               # fases por perfil
OUTPUT_DIR = "profiles" # carpeta donde guardar

os.makedirs(OUTPUT_DIR, exist_ok=True)

# posibles valores de CPU: 0.1, 0.2, ..., 1.0
cpu_vals = [round(x * 0.1, 1) for x in range(1, 7)]

for cliente_id in range(1, N_CLIENTES + 1):
    random.seed(42 + cliente_id)  # semilla única por cliente

    perfil = []
    for _ in range(FASES):
        fase = {
            "duracion": random.randint(500, 1000),
            "cpus": random.choice(cpu_vals),
        }
        perfil.append(fase)

    # guardar en archivo YAML
    filename = os.path.join(OUTPUT_DIR, f"client_{cliente_id}.yaml")
    with open(filename, "w") as f:
        yaml.dump({"perfil": perfil}, f)

    print(f"Perfil generado para cliente {cliente_id}: {filename}")
