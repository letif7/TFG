from .all_terms import optimizar_plegamiento_proteina
"""
def main():
    """"""Función principal que ejecuta la optimización.""""""
    try:
        print("Iniciando optimización de plegamiento de proteínas...")
        
        # Ejecutar con parámetros por defecto
        solutions, stats = optimizar_plegamiento_proteina()
        
        print(f"\nOptimización completada exitosamente!")
        print(f"Se encontraron {len(solutions)} soluciones no dominadas")
        
    except Exception as e:
        print(f"Error durante la optimización: {e}")
        return 1
    
    return 0
"""
import matplotlib
matplotlib.use("Agg")  # <- esto debe ir *antes* de importar pyplot
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import pandas as pd

def export_solution_to_pdb(var_file, solution_index, pdb_output):
    """
    Convierte una solución de VAR.txt en un archivo .pdb
    
    :param var_file: Ruta al archivo VAR.txt
    :param solution_index: Índice (fila) de la solución que quieres exportar
    :param pdb_output: Nombre de archivo .pdb de salida
    """
    # Cargar todas las soluciones
    data = np.loadtxt(var_file)
    
    # Seleccionar la solución deseada (fila)
    coords = data[solution_index]
    
    # Asegurar que sea múltiplo de 3 (x, y, z)
    assert len(coords) % 3 == 0, "Número de variables no es múltiplo de 3"
    num_atoms = len(coords) // 3
    
    # Crear archivo PDB
    with open(pdb_output, "w") as f:
        for i in range(num_atoms):
            x, y, z = coords[3*i:3*i+3]
            atom_line = (
                f"ATOM  {i+1:5d}  CA  UNK A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            f.write(atom_line)
        f.write("END\n")
    
    print(f"Solución {solution_index} exportada a {pdb_output}")

def generar_tabla_resultados(results_dir="/home/lnfg/TFG/KCM_NSGAII/results"):
    # Buscar el último archivo objectives y variables
    obj_files = glob.glob(os.path.join(results_dir, "objectives_*.txt"))
    var_files = glob.glob(os.path.join(results_dir, "variables_*.txt"))

    if not obj_files or not var_files:
        print("No se encontraron archivos de resultados.")
        return

    latest_obj = max(obj_files, key=os.path.getctime)
    latest_var = max(var_files, key=os.path.getctime)

    print(f"Usando archivos:\n  {latest_obj}\n  {latest_var}")

    # Leer datos
    objectives = np.loadtxt(latest_obj)
    variables = np.loadtxt(latest_var)

    # Construir tabla (cada fila de objectives corresponde a misma fila de variables)
    df = pd.DataFrame({
        "RMSD": objectives[:,0],
        "GDT": -objectives[:,1],   # lo pones en positivo si lo guardaste negado
        "Energía": objectives[:,2],
        "Fila_vars": range(len(variables))  # índice que corresponde a variables[i]
    })

    # Guardar a CSV
    output_csv = os.path.join(results_dir, "tabla_resultados.csv")
    df.to_csv(output_csv, index=False)
    print(f"Tabla guardada en: {output_csv}")

    return df

def plot_latest_objectives():
    # Carpeta donde se guardan los resultados
    #results_dir = "/home/lnfg/TFG/KCM_NSGAII/results"
    results_dir = "/home/ubuntu/results"
    print(results_dir)

    # Buscar el último archivo objectives_*.txt
    files = glob.glob(os.path.join(results_dir, "objectives_*.txt"))
    if not files:
        print(files)
        print("No se encontraron archivos objectives_*.txt en la carpeta 'results'")
        return
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Usando archivo: {latest_file}")
    
    # Leer los datos
    data = np.loadtxt(latest_file)
    if data.shape[1] < 3:
        print("El archivo no contiene 3 columnas de objetivos.")
        return
    
    # Graficar en 3D
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter(data[:,0], data[:,1], data[:,2], c="blue", marker="o", alpha=0.7)
    ax.set_xlabel("Objetivo 1 (RMSD)")
    ax.set_ylabel("Objetivo 2 (-GDT)")
    ax.set_zlabel("Objetivo 3 (Energía)")
    ax.set_title("Frente de Pareto aproximado")
    
    # Guardar la figura a archivo
    output_path = os.path.join(results_dir, "grafico.png")
    plt.savefig(output_path, dpi=300)
    print(f"Gráfico guardado en: {output_path}")

def exportar_pdb_mejorado(variables, output_pdb, secuencia_aminoacidos=None):
    """
    Exporta una solución a PDB con mejor información de conectividad.
    
    Args:
        variables: Array de coordenadas [x1, y1, z1, x2, y2, z2, ...]
        output_pdb: Archivo PDB de salida
        secuencia_aminoacidos: Secuencia de aminoácidos (opcional)
    """
    
    # Verificar que sea múltiplo de 3
    if len(variables) % 3 != 0:
        print(f"❌ Error: Variables no son múltiplo de 3")
        return
    
    # Convertir a coordenadas 3D
    coords = variables.reshape(-1, 3)
    num_atomos = len(coords)
    
    # Estimar número de residuos (asumiendo 4 átomos por residuo)
    num_residuos = num_atomos // 4 if num_atomos % 4 == 0 else num_atomos
    
    print(f"🧬 Exportando PDB mejorado:")
    print(f"   └─ Átomos: {num_atomos}")
    print(f"   └─ Residuos estimados: {num_residuos}")
    print(f"   └─ Archivo: {output_pdb}")
    
    # Nombres de átomos típicos del backbone
    atom_names = ['N', 'CA', 'C', 'O'] if num_atomos % 4 == 0 else ['CA']
    
    with open(output_pdb, 'w') as f:
        # Header mejorado
        f.write("HEADER    PROTEIN FOLDING NSGA-II OPTIMIZATION\n")
        f.write("TITLE     OPTIMIZED PROTEIN STRUCTURE\n")
        f.write("REMARK   Generated from NSGA-II optimization variables\n")
        f.write("REMARK   Each residue represented by backbone atoms\n")
        f.write("MODEL        1\n")
        
        # Escribir átomos con mejor información
        atom_counter = 1
        residue_counter = 1
        
        for i, (x, y, z) in enumerate(coords):
            # Determinar tipo de átomo y residuo
            if num_atomos % 4 == 0:
                # 4 átomos por residuo
                atom_idx = i % 4
                if atom_idx == 0:
                    residue_counter = (i // 4) + 1
                atom_name = atom_names[atom_idx]
            else:
                # Solo CA
                atom_name = 'CA'
                residue_counter = i + 1
            
            # Determinar residuo (usar secuencia si está disponible)
            if secuencia_aminoacidos and residue_counter <= len(secuencia_aminoacidos):
                res_name = secuencia_aminoacidos[residue_counter - 1]
            else:
                res_name = 'GLY'  # Glicina por defecto
            
            # Elemento químico basado en el nombre del átomo
            element = atom_name[0] if atom_name[0] in ['N', 'C', 'O'] else 'C'
            
            # Escribir línea ATOM en formato PDB estándar
            atom_line = (
                f"ATOM  {atom_counter:5d}  {atom_name:<4s} {res_name} A{residue_counter:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element}\n"
            )
            f.write(atom_line)
            atom_counter += 1
        
        # Agregar información de conectividad para backbone
        if num_atomos % 4 == 0:
            f.write("REMARK   CONNECTIVITY INFORMATION\n")
            for i in range(0, num_atomos - 4, 4):
                # Conectar C de residuo i con N de residuo i+1
                c_atom = i + 3  # C es el 4to átomo (índice 3)
                n_next = i + 4 + 1  # N del siguiente residuo (índice 0) + 1 para 1-based
                if c_atom < num_atomos and n_next <= num_atomos:
                    f.write(f"CONECT{c_atom+1:5d}{n_next:5d}\n")
        
        f.write("ENDMDL\n")
        f.write("END\n")
    
    print(f"✅ PDB mejorado exportado")
    return output_pdb

def crear_script_pymol(pdb_file, output_script="visualizar.pml"):
    """
    Crea un script de PyMOL para visualizar mejor la estructura.
    
    Args:
        pdb_file: Archivo PDB a visualizar
        output_script: Script de PyMOL de salida
    """
    
    script_content = f'''# Script de PyMOL para visualizar estructura optimizada
# Uso: pymol -c {output_script}

# Cargar estructura
load {pdb_file}, protein

# Configuración básica
bg_color white
set_view [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -50.0, 100.0, -100.0]

# Estilo de visualización mejorado
hide everything
show spheres, protein
color green, protein
set sphere_scale, 0.3

# Mostrar cadena conectada si es posible
show sticks, protein
color yellow, protein and name CA
set stick_radius, 0.1

# Conectar átomos secuencialmente para simular backbone
distance chain_connectivity, protein and name CA, protein and name CA, cutoff=8.0, mode=0
hide labels, chain_connectivity
color red, chain_connectivity
set dash_width, 2.0

# Etiquetas informativas
label protein and name CA, "CA"
set label_size, 8

# Vista centrada
center protein
zoom protein

# Información en consola
print "Estructura cargada desde optimización NSGA-II"
print "Verde: Átomos de la proteína"
print "Rojo: Conexiones estimadas del backbone"
print "Usa 'show cartoon' si la conectividad se ve bien"

# Comandos útiles para el usuario
print "Comandos útiles:"
print "  show cartoon          - Mostrar como ribbon/cartoon"
print "  show surface          - Mostrar superficie"
print "  color rainbow         - Colorear por secuencia"
print "  set transparency, 0.5 - Hacer transparente"
'''
    
    with open(output_script, 'w') as f:
        f.write(script_content)
    
    print(f"📝 Script de PyMOL creado: {output_script}")
    print(f"💡 Uso: pymol {output_script}")
    return output_script

def comandos_pymol_consola():
    """
    Imprime comandos útiles para usar en la consola de PyMOL.
    """
    print("🎨 COMANDOS PARA PYMOL CONSOLA:")
    print("="*50)
    print("# 1. Cambiar representación:")
    print("hide everything")
    print("show spheres")
    print("set sphere_scale, 0.5")
    print()
    print("# 2. Mostrar como cadena conectada:")
    print("show sticks")
    print("set stick_radius, 0.2")
    print()
    print("# 3. Conectar átomos cercanos (simular backbone):")
    print("distance connections, all, all, cutoff=4.0, mode=0")
    print("hide labels, connections")
    print("color red, connections")
    print()
    print("# 4. Si tienes suerte y la estructura es coherente:")
    print("show cartoon")
    print("color rainbow")
    print()
    print("# 5. Centrar y hacer zoom:")
    print("center all")
    print("zoom all")
    print()
    print("# 6. Cambiar colores:")
    print("color green")
    print("color_by_rainbow")
    print()
    print("# 7. Guardar imagen:")
    print("png estructura_optimizada.png, dpi=300")

def analizar_calidad_estructura(variables, umbral_distancia=15.0):
    """
    Analiza la calidad de la estructura optimizada.
    
    Args:
        variables: Coordenadas de la estructura
        umbral_distancia: Distancia máxima razonable entre átomos consecutivos
    
    Returns:
        dict: Análisis de calidad
    """
    coords = variables.reshape(-1, 3)
    
    # Calcular distancias entre átomos consecutivos
    distancias = []
    for i in range(len(coords) - 1):
        dist = np.linalg.norm(coords[i+1] - coords[i])
        distancias.append(dist)
    
    distancias = np.array(distancias)
    
    # Análisis estadístico
    analisis = {
        'num_atomos': len(coords),
        'distancia_promedio': np.mean(distancias),
        'distancia_min': np.min(distancias),
        'distancia_max': np.max(distancias),
        'distancia_std': np.std(distancias),
        'distancias_razonables': np.sum(distancias < umbral_distancia),
        'porcentaje_razonable': (np.sum(distancias < umbral_distancia) / len(distancias)) * 100
    }
    
    # Evaluación de compacidad
    centro_masa = np.mean(coords, axis=0)
    distancias_al_centro = [np.linalg.norm(coord - centro_masa) for coord in coords]
    analisis['radio_giro'] = np.sqrt(np.mean(np.array(distancias_al_centro)**2))
    
    # Dimensiones de la caja delimitadora
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    dimensiones = max_coords - min_coords
    analisis['dimensiones_caja'] = dimensiones
    analisis['volumen_caja'] = np.prod(dimensiones)
    
    return analisis

def imprimir_analisis_calidad(analisis):
    """Imprime análisis de calidad de forma legible."""
    print("🔍 ANÁLISIS DE CALIDAD DE LA ESTRUCTURA:")
    print("="*50)
    print(f"📊 Estadísticas básicas:")
    print(f"   └─ Número de átomos: {analisis['num_atomos']}")
    print(f"   └─ Radio de giro: {analisis['radio_giro']:.2f} Å")
    print(f"   └─ Dimensiones: {analisis['dimensiones_caja'][0]:.1f} × {analisis['dimensiones_caja'][1]:.1f} × {analisis['dimensiones_caja'][2]:.1f} Å")
    print()
    print(f"🔗 Conectividad:")
    print(f"   └─ Distancia promedio entre átomos: {analisis['distancia_promedio']:.2f} Å")
    print(f"   └─ Distancia mínima: {analisis['distancia_min']:.2f} Å")
    print(f"   └─ Distancia máxima: {analisis['distancia_max']:.2f} Å")
    print(f"   └─ Desviación estándar: {analisis['distancia_std']:.2f} Å")
    print(f"   └─ Conexiones razonables: {analisis['porcentaje_razonable']:.1f}%")
    print()
    
    # Interpretación
    if analisis['porcentaje_razonable'] > 80:
        print("✅ Estructura parece coherente (>80% conexiones razonables)")
    elif analisis['porcentaje_razonable'] > 50:
        print("⚠️ Estructura parcialmente coherente (50-80% conexiones razonables)")
    else:
        print("❌ Estructura muy dispersa (<50% conexiones razonables)")
    
    if analisis['radio_giro'] < 50:
        print("✅ Estructura compacta (radio de giro < 50 Å)")
    elif analisis['radio_giro'] < 100:
        print("⚠️ Estructura moderadamente extendida (50-100 Å)")
    else:
        print("❌ Estructura muy extendida (>100 Å)")

# Función principal integrada
def mejorar_visualizacion_completa(variables_file, solucion_numero=1, 
                                 secuencia_aa=None):
    """
    Función completa para mejorar la visualización de una solución.
    
    Args:
        variables_file: Archivo de variables
        solucion_numero: Número de solución a exportar
        secuencia_aa: Secuencia de aminoácidos (opcional)
    """
    
    print("🎯 MEJORANDO VISUALIZACIÓN DE ESTRUCTURA OPTIMIZADA")
    print("="*60)
    
    # Cargar variables
    variables = np.loadtxt(variables_file)
    
    if solucion_numero < 1 or solucion_numero > len(variables):
        print(f"❌ Error: Solución {solucion_numero} no existe")
        return
    
    # Obtener coordenadas de la solución
    coords = variables[solucion_numero - 1]
    
    # Analizar calidad
    analisis = analizar_calidad_estructura(coords)
    imprimir_analisis_calidad(analisis)
    
    # Exportar PDB mejorado
    pdb_file = f"solucion_{solucion_numero}_mejorada.pdb"
    exportar_pdb_mejorado(coords, pdb_file, secuencia_aa)
    
    # Crear script de PyMOL
    script_file = f"visualizar_solucion_{solucion_numero}.pml"
    crear_script_pymol(pdb_file, script_file)
    
    # Mostrar comandos útiles
    print()
    comandos_pymol_consola()
    
    print(f"\n🎉 ARCHIVOS GENERADOS:")
    print(f"   └─ PDB mejorado: {pdb_file}")
    print(f"   └─ Script PyMOL: {script_file}")
    print(f"\n💡 CÓMO USAR:")
    print(f"   1. Abrir PyMOL")
    print(f"   2. File → Run Script → {script_file}")
    print(f"   3. O usar: pymol {script_file}")
    
    return pdb_file, script_file

def exportar_solucion_especifica(variables_file, solucion_numero, output_pdb):
    """
    Exporta una solución específica a archivo PDB para visualización.
    
    Args:
        variables_file: Archivo de variables
        solucion_numero: Número de solución (1-indexado) 
        output_pdb: Archivo PDB de salida
    """
    
    # Cargar variables
    variables = np.loadtxt(variables_file)
    
    # Verificar que la solución existe
    if solucion_numero < 1 or solucion_numero > len(variables):
        print(f"❌ Error: Solución {solucion_numero} no existe")
        print(f"   Soluciones disponibles: 1-{len(variables)}")
        return
    
    # Obtener coordenadas de la solución (convertir a 0-indexado)
    coords = variables[solucion_numero - 1]
    
    # Verificar que es múltiplo de 3
    if len(coords) % 3 != 0:
        print(f"❌ Error: Variables no son múltiplo de 3")
        return
    
    # Convertir a coordenadas 3D
    coords_3d = coords.reshape(-1, 3)
    num_atomos = len(coords_3d)
    
    print(f"🧬 Exportando solución {solucion_numero}:")
    print(f"   └─ Número de átomos: {num_atomos}")
    print(f"   └─ Archivo PDB: {output_pdb}")
    
    # Escribir archivo PDB
    with open(output_pdb, 'w') as f:
        f.write(f"HEADER    PROTEIN FOLDING SOLUTION {solucion_numero}\n")
        f.write(f"TITLE     OPTIMIZED STRUCTURE\n")
        f.write(f"MODEL        1\n")
        
        for i, (x, y, z) in enumerate(coords_3d):
            # Formato PDB estándar
            atom_line = (
                f"ATOM  {i+1:5d}  CA  UNK A{i//4+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            f.write(atom_line)
        
        f.write("ENDMDL\n")
        f.write("END\n")
    
    print(f"✅ Solución exportada exitosamente")


def crear_tabla_correspondencia(objectives_file, variables_file, output_csv=None):
    """
    Crea una tabla que muestra la correspondencia clara entre objetivos y variables.
    
    Args:
        objectives_file: Archivo de objetivos
        variables_file: Archivo de variables  
        output_csv: Archivo CSV de salida (opcional)
    """
    
    # Cargar datos
    objetivos = np.loadtxt(objectives_file)
    variables = np.loadtxt(variables_file)
    
    # Crear DataFrame
    data = []
    for i in range(len(objetivos)):
        row = {
            'solucion_id': i + 1,
            'rmsd': objetivos[i][0],
            'gdt_negativo': objetivos[i][1],
            'gdt_real': -objetivos[i][1],  # GDT real (positivo)
            'energia': objetivos[i][2],
            'num_variables': len(variables[i]),
            'primera_coord_x': variables[i][0],
            'primera_coord_y': variables[i][1], 
            'primera_coord_z': variables[i][2],
            'linea_objetivos': i + 1,
            'linea_variables': i + 1
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Guardar si se especifica archivo
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"📁 Tabla guardada en: {output_csv}")
    
    return df


def export_solution_to_pdb_4atoms(variables, output_pdb):
    """
    Exporta una solución con 4 átomos por residuo (N, CA, C, O).
    
    Args:
        variables: Array de variables [x1,y1,z1,x2,y2,z2,...] 
        output_pdb: Nombre del archivo PDB de salida
    """
    
    # Verificar que sea múltiplo de 12 (4 átomos × 3 coordenadas)
    if len(variables) % 12 != 0:
        print(f"Error: Variables ({len(variables)}) no son múltiplo de 12")
        return
    
    # Calcular número de residuos
    num_residues = len(variables) // 12
    atom_names = ['N', 'CA', 'C', 'O']
    
    with open(output_pdb, 'w') as f:
        f.write("HEADER    NSGA-II PROTEIN OPTIMIZATION\n")
        f.write("MODEL        1\n")
        
        atom_counter = 1
        
        # Para cada residuo
        for residue_idx in range(num_residues):
            
            # Para cada átomo del residuo (N, CA, C, O)
            for atom_idx in range(4):
                
                # Calcular índice en el array de variables
                var_idx = (residue_idx * 4 + atom_idx) * 3
                
                # Extraer coordenadas x, y, z
                x = variables[var_idx]
                y = variables[var_idx + 1]
                z = variables[var_idx + 2]
                
                # Nombre del átomo
                atom_name = atom_names[atom_idx]
                
                # Escribir línea PDB
                f.write(f"ATOM  {atom_counter:5d}  {atom_name:<4s}GLY A{residue_idx+1:4d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {atom_name[0]}\n")
                
                atom_counter += 1
        
        f.write("ENDMDL\n")
        f.write("END\n")
    
    print(f"Exportado: {output_pdb} con {num_residues} residuos completos")

def export_best_solutions_4atoms(variables_file, n_solutions=5):
    """
    Exporta las primeras n soluciones con 4 átomos por residuo.
    
    Args:
        variables_file: Archivo de variables
        n_solutions: Número de soluciones a exportar
    """
    
    # Cargar todas las soluciones
    data = np.loadtxt(variables_file)
    
    print(f"Exportando {min(n_solutions, len(data))} soluciones...")
    
    # Exportar cada solución
    for i in range(min(n_solutions, len(data))):
        output_file = f"solution_{i+1:03d}_4atoms.pdb"
        export_solution_to_pdb_4atoms(data[i], output_file)
    
    print("¡Exportación completada!")


import subprocess, tempfile, os, sys, pandas as pd, textwrap, shutil
from pathlib import Path

def _ensure_esm_examples_available():
    """
    Verifica que los módulos de ejemplo de ESM (esm.examples.inverse_folding.*) existan.
    Si no existen, explica cómo instalar el repo completo (no solo fair-esm de PyPI).
    """
    try:
        import importlib
        importlib.import_module("esm.examples.inverse_folding.sample_sequences")
        importlib.import_module("esm.examples.inverse_folding.score_log_likelihoods")
        return True
    except Exception as e:
        msg = textwrap.dedent(f"""
        [ESM] No se encontraron los módulos de ejemplo:
              esm.examples.inverse_folding.sample_sequences / score_log_likelihoods

        ➜ Solución recomendada (una sola vez):
            pip uninstall -y fair-esm esm
            pip install git+https://github.com/facebookresearch/esm.git

        Motivo: el paquete de PyPI 'fair-esm' no incluye los scripts 'esm/examples/...'.
        El repo de GitHub sí los trae y son los que usamos para muestrear y puntuar secuencias.

        Error original: {e}
        """)
        print(msg)
        return False


def esm_if_design_and_score(pdb_path: str,
                            chain: str = "A",
                            num_samples: int = 10,
                            temperature: float = 1e-6,
                            workdir: str = None,
                            esm_repo_dir: str = "/home/lnfg/TFG/KCM_NSGAII/third_party/esm"):
    """
    Diseña secuencias con ESM-IF1 y las puntúa llamando a los scripts del repo clonado.
    Requisitos:
      - Haber hecho: git clone https://github.com/facebookresearch/esm.git  (ver ruta en esm_repo_dir)
      - Tener torch instalado en tu entorno.
    Devuelve: (ruta_fasta, ruta_csv_scores, dict_fila_mejor)
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB no encontrado: {pdb_path}")

    sample_py = os.path.join(esm_repo_dir, "examples", "inverse_folding", "sample_sequences.py")
    score_py  = os.path.join(esm_repo_dir, "examples", "inverse_folding", "score_log_likelihoods.py")
    if not (os.path.isfile(sample_py) and os.path.isfile(score_py)):
        raise FileNotFoundError(
            "No se encontraron los scripts de inverse folding en el repo ESM.\n"
            f"Esperado:\n  {sample_py}\n  {score_py}\n"
            "Solución: verifica que clonaste correctamente el repo y que esm_repo_dir apunta bien."
        )

    tmp = workdir or tempfile.mkdtemp(prefix="esm_if_")
    Path(tmp).mkdir(parents=True, exist_ok=True)
    fasta_out  = os.path.join(tmp, "designs.fasta")
    scores_out = os.path.join(tmp, "scores.csv")

    # 1) muestrear
    cmd_sample = [
        sys.executable, sample_py,
        pdb_path, "--chain", chain,
        "--temperature", str(temperature),
        "--num-samples", str(num_samples),
        "--outpath", fasta_out,
    ]
    print(f"[ESM] Muestreando {num_samples} secuencias → {fasta_out}")
    subprocess.run(cmd_sample, check=True)

    # 2) puntuar
    cmd_score = [
        sys.executable, score_py,
        pdb_path, fasta_out, "--chain", chain,
        "--outpath", scores_out,
    ]
    print(f"[ESM] Calculando log-likelihoods → {scores_out}")
    subprocess.run(cmd_score, check=True)

    df = pd.read_csv(scores_out)
    if "avg_log_likelihood" not in df.columns:
        raise RuntimeError(f"[ESM] CSV inesperado. Columnas: {list(df.columns)}")
    best = df.sort_values("avg_log_likelihood", ascending=False).iloc[0].to_dict()
    return fasta_out, scores_out, best

if __name__ == "__main__":
    #plot_latest_objectives()
    #tabla = generar_tabla_resultados()
    #print(tabla.head())
    #/home/lnfg/TFG/KCM_NSGAII/results/variables_20250811_210807.txt
    #export_solution_to_pdb("/home/lnfg/TFG/KCM_NSGAII/results/variables_20250811_210807.txt", solution_index=1, pdb_output="solucion_2.pdb")
    
    #tabla = crear_tabla_correspondencia("/home/lnfg/TFG/KCM_NSGAII/results/objectives_20250811_210807.txt", "/home/lnfg/TFG/KCM_NSGAII/results/variables_20250811_210807.txt", "correspondencia.csv")
    #print("\n📊 TABLA DE CORRESPONDENCIA (primeras 5 filas):")
    #print(tabla.head().to_string(index=False))

    try:
        backbone_pdb = "/home/lnfg/TFG/KCM_NSGAII/results/mejor_solucion_4atoms.pdb"
        out_fasta, out_csv, best_row = esm_if_design_and_score(
            pdb_path=backbone_pdb,
            chain="A",
            num_samples=10,
            temperature=1e-6,
            workdir="/home/lnfg/TFG/KCM_NSGAII/results",   # carpeta donde guardar
            esm_repo_dir="/home/lnfg/TFG/KCM_NSGAII/third_party/esm"
        )
        print("\n🧬 RESULTADOS ESM-IF1")
        print(f"  FASTA: {out_fasta}")
        print(f"  SCORES: {out_csv}")
        print(f"  Top avg_log_likelihood: {best_row.get('avg_log_likelihood')}")
        print(f"  Best sequence: {best_row.get('sequence')}")
    except Exception as e:
        print(f"[ESM] Error en diseño de secuencias: {e}")

    """ try:
        print("Iniciando optimización de plegamiento de proteínas...")
        
        # Ejecutar con parámetros por defecto
        solutions, stats = optimizar_plegamiento_proteina()
        
        print(f"\nOptimización completada exitosamente!")
        print(f"Se encontraron {len(solutions)} soluciones no dominadas")
        
    except Exception as e:
        print(f"Error durante la optimización: {e}")
 """
    # Exportar mejor solución (menor RMSD)
    #mejor_solucion = tabla.loc[tabla['rmsd'].idxmin()]
    #print(f"\n🏆 MEJOR SOLUCIÓN (menor RMSD):")
    #print(f"   └─ Solución #{int(mejor_solucion['solucion_id'])}")
    #print(f"   └─ RMSD: {mejor_solucion['rmsd']:.6f}")
    #print(f"   └─ GDT: {mejor_solucion['gdt_real']:.6f}")
    #print(f"   └─ Energía: {mejor_solucion['energia']:.2f}")

    #exportar_solucion_especifica("/home/lnfg/TFG/KCM_NSGAII/results/variables_20250811_210807.txt", int(mejor_solucion['solucion_id']), "mejor_solucion.pdb")
    #mejorar_visualizacion_completa(
    #    "/home/lnfg/TFG/KCM_NSGAII/results/variables_20250811_210807.txt", 
    #    solucion_numero=1,
    #    secuencia_aa=None  # Cambiar por tu secuencia si la tienes
    #)
        # Exportar primera solución
    #data = np.loadtxt("/home/lnfg/TFG/KCM_NSGAII/results/variables_20250811_210807.txt")
    #export_solution_to_pdb_4atoms(data[0], "mejor_solucion_4atoms.pdb")