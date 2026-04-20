import os
import glob
from pathlib import Path

def clean_out_files(base_dir, start_num=0):
    """
    Limpia archivos *.out en base_dir:
    - Toma el último *.out (mayor número)
    - Recorre desde start_num hasta ese número
    - Filtra líneas: mantiene solo las que tienen "INFO" y NO tienen "Fleet"
    """
    base_dir = Path(base_dir)
    
    # Encontrar el mayor número en *.out
    out_files = list(base_dir.glob('*.out'))
    if not out_files:
        print(f"❌ No hay archivos *.out en {base_dir}")
        return
    
    # Extraer números y encontrar máximo
    numbers = []
    for f in out_files:
        num_match = f.stem.rstrip('out')
        if num_match.isdigit():
            numbers.append(int(num_match))
    
    if not numbers:
        print(f"❌ No se encontraron números en nombres *.out en {base_dir}")
        return
    
    max_num = max(numbers)
    print(f"📁 Directorio: {base_dir}")
    print(f"🔢 Archivos encontrados: 0 → {max_num}")
    
    processed = 0
    for num in range(start_num, max_num + 1):
        input_file = base_dir / f"{num}.out"
        
        if not input_file.exists():
            print(f"⚠️  Saltando {num}.out (no existe)")
            continue
        
        output_file = base_dir / f"{num}_cleaned.out"
        
        # Leer y filtrar
        with open(input_file, "r", encoding="utf-8") as f:
            lineas = [linea for linea in f if "INFO" in linea and "Fleet" not in linea]
        
        # Escribir archivo limpio
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(lineas)
        
        processed += 1
        print(f"✅ {num}.out → {num}_cleaned.out ({len(lineas)} líneas)")
    
    print(f"\n🎉 Procesados {processed}/{max_num+1-start_num} archivos")

# ==============================
# USO
# ==============================
base_directory = "/home/usuario/Escritorio/FL-Semiasync/local-execution/results/FedMOpt/1062781"  # ← CAMBIA ESTO
clean_out_files(base_directory, start_num=0)