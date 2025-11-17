import pandas as pd

# Datos
data = {
    'Año': [2020, 2021, 2022, 2023, 2024, 2025],
    'Artículos': [1, 3, 3, 10, 14, 20]
}
df = pd.DataFrame(data)

# Valores inicial y final (desde 2022)
N_inicio = df.loc[df['Año'] == 2022, 'Artículos'].values[0]
N_final = df.loc[df['Año'] == 2025, 'Artículos'].values[0]

# Número de años transcurridos
años = 2025 - 2022

# Calcular CAGR
CAGR = ((N_final / N_inicio) ** (1 / años) - 1) * 100

print(f"Tasa media de crecimiento anual (CAGR) desde 2022: {CAGR:.2f}%")
