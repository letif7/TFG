import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import esm
import pyrosetta
import shutil

# jMetalPy imports corregidos
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.observer import ProgressBarObserver, PrintObjectivesObserver, BasicObserver
# BioPython

from Bio.PDB import PDBParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter("ignore", PDBConstructionWarning)

from jmetal.util.observer import Observer

class GuardarParetoCadaN(Observer):

    def __init__(self, algorithm, cada=10, drive_dir="/content/drive/MyDrive/resultados_proteina"):
        self.algorithm = algorithm
        self.cada = cada
        self.drive_dir = drive_dir
        os.makedirs("partial_results", exist_ok=True)  # ← faltaba esto
        os.makedirs(drive_dir, exist_ok=True)

    def update(self, *args, **kwargs):
        evaluaciones = kwargs.get("EVALUATIONS", 0)
        solutions = kwargs.get("SOLUTIONS", [])

        # ← Agregá este print para confirmar que se llama
        print(f"DEBUG observer: eval={evaluaciones}, sols={len(solutions) if solutions else 0}")

        if evaluaciones % self.cada == 0 and evaluaciones > 0 and solutions:
            try:
                pareto = get_non_dominated_solutions(solutions)
                if not pareto:
                    return

                carpeta_local = f"partial_results/eval_{evaluaciones}"
                os.makedirs(carpeta_local, exist_ok=True)

                print_function_values_to_file(pareto, os.path.join(carpeta_local, "PARETO_FUN.tsv"))
                print_variables_to_file(pareto, os.path.join(carpeta_local, "PARETO_VAR.tsv"))
                exportar_pareto_y_variables_csv(pareto, carpeta_local)
                plot_pareto_front_3d(pareto, carpeta_local)

                carpeta_drive = os.path.join(self.drive_dir, f"checkpoint_eval_{evaluaciones}")
                shutil.copytree(carpeta_local, carpeta_drive, dirs_exist_ok=True)

                print(f"Checkpoint guardado en Drive: eval {evaluaciones} (pareto={len(pareto)})")

            except Exception as e:
                print(f"Error guardando checkpoint: {e}")
                import traceback
                traceback.print_exc()  # ← muestra el error completo

                
# Importar tus clases locales
from .ProteinFoldingProblem.ProteinFoldingProblem import ProteinFoldingProblem
from .pdb_seq_tools import extract_amino_acid_sequence, det_sec, extract_backbone_atoms_str
from .Algorithm_evolutionary.algorithm_evolutionary import EDA_isla

from .utils import (
    StoppingByTime,
    StoppingByDiversity,
    CombinedTermination
)

# --- Función principal de optimización ---
def optimizar_plegamiento_proteina(
    pdb_reference_file: str = None,
    max_evaluations: int = 100,
    population_size: int = 100,
    use_physicochemical_descriptors: bool = True,
    corte: list = [1.0, 2.0, 4.0, 8.0],
    energia_params: tuple = (-50.0, 10.0),
    crossover_probability: float = 1,
    mutation_probability: float = 0.05,
    crossover_distribution_index: float = 20.0,
    mutation_distribution_index: float = 20.0,
    output_dir: str = "results",
    verbose: bool = True
) -> tuple:
    
    # Directorio de trabajo en Colab
    directorio_trabajo = os.getcwd()

    # Obtener PDB
    if pdb_reference_file is None:
        pdb_reference_file = obtener_primer_pdb(input_folder='demo_pdbs')
    if not os.path.exists(pdb_reference_file):
        raise FileNotFoundError(f"No se encontró el archivo PDB: {pdb_reference_file}")

    pdb_content = leer_archivo_pdb(pdb_reference_file)

    # Extraer secuencia de aminoácidos
    amino_seq = extract_amino_acid_sequence(pdb_content)
    num_residues = len(amino_seq)

    pdb_file = pdb_reference_file  # Usamos directamente la referencia

    # Crear problema
    problem = ProteinFoldingProblem(
        pdb_reference=pdb_file,
        sequence_length=num_residues,
        use_physicochemical_descriptors=use_physicochemical_descriptors,
        corte=corte,
        energia_params=energia_params
    )
    num_objectives = problem.number_of_objectives()  # = 3


    # Operadores genéticos
    crossover = SBXCrossover(probability=crossover_probability, distribution_index=crossover_distribution_index)
    mutation = PolynomialMutation(probability=mutation_probability, distribution_index=mutation_distribution_index)
    # --- Criterio 1: límite de 24 horas ---
    termination_time = StoppingByTime(
        max_seconds=24 * 60 * 60
        #x_seconds=780
    )

    # --- Criterio 2: convergencia por diversidad ---
    termination_div = StoppingByDiversity(
        min_diversity=0.12,      # ajustar experimentalmente
        patience=25,             # generaciones consecutivas
        amino_seq_ref=amino_seq
    )

    # --- Criterio combinado ---
    termination = CombinedTermination([
        termination_time,
        termination_div
    ])

# --- Algoritmo NSGA-II ---
    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=termination
    )
   

    # Barra de progreso (tqdm) hasta max_evaluations
    algorithm.observable.register(observer=ProgressBarObserver(max_evaluations))

    # Log cada N evaluaciones (elige uno)
    #algorithm.observable.register(observer=BasicObserver(frequency=10))
    algorithm.observable.register(observer=GuardarParetoCadaN(
        algorithm=algorithm,
        cada=population_size,# generación completa
        drive_dir="/content/drive/MyDrive/resultados_proteina"
    ))
    # o si solo querés imprimir fitness:
    # algorithm.observable.register(observer=PrintObjectivesObserver(frequency=10))

    algorithm.run()
    print(f"Termino de correr el algoritmo")
    solutions = algorithm.result()

    # Validar que objectives son 7 floats
    for i, s in enumerate(solutions):
        if len(s.objectives) != num_objectives:
            raise ValueError(f"Sol #{i}: len(objectives)={len(s.objectives)} != {num_objectives}")
        for j, v in enumerate(s.objectives):
            try:
                float(v)
            except Exception:
                raise TypeError(f"Sol #{i} obj#{j} no convertible a float: {type(v)} -> {v}")

    non_dominated_solutions = get_non_dominated_solutions(solutions)

    # Mostrar resultados
    mostrar_top_soluciones(non_dominated_solutions, top_n=len(non_dominated_solutions))

    # Crear carpeta de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Guardar archivos estándar jMetal
    print_function_values_to_file(non_dominated_solutions, os.path.join(output_dir, "PARETO_FUN.tsv"))
    print_variables_to_file(non_dominated_solutions, os.path.join(output_dir, "PARETO_VAR.tsv"))

    # Guardar CSV legible + gráfico
    exportar_pareto_y_variables_csv(non_dominated_solutions, output_dir)
    plot_pareto_front_3d(non_dominated_solutions, output_dir)

    # Elegir mejor solución
    best = elegir_mejor_solucion(non_dominated_solutions, prefer='rmsd')

    # Obtener secuencia de la mejor solución
    attrs = getattr(best, "attributes", {}) or {}
    best_seq = attrs.get("sequence") or attrs.get("amino_sequence")
    if not best_seq:
        posiciones = [int(round(v)) for v in best.variables]
        best_seq = det_sec(posiciones, amino_seq)

    # Guardar PDB con ESMFold
    esm_pdb_path = os.path.join(output_dir, "best_by_fitness_esmfold.pdb")
    try:
        ruta = guardar_pdb_con_esmfold(best_seq, esm_pdb_path)
        print(f"PDB (ESMFold) guardada en: {ruta}")
    except Exception as e:
        print(f"⚠️ No se pudo generar PDB con ESMFold: {e}")

    # Mapear floats a aminoácidos
    for sol in non_dominated_solutions:
        posiciones = [int(round(v)) for v in sol.variables]
        sol.attributes['amino_sequence'] = det_sec(posiciones, amino_seq)

    return non_dominated_solutions, crear_estadisticas_detalladas(non_dominated_solutions, 0, max_evaluations, population_size)


# --- Función auxiliar para ESMFold ---
def guardar_pdb_con_esmfold(sequence: str, out_path: str, device: str = None, chunk_size: int = 64):
    import torch
    import esm

    seq = (sequence or "").strip().upper().replace(" ", "")
    if not seq:
        raise ValueError("Secuencia vacía para ESMFold.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    try:
        model.set_chunk_size(chunk_size)
    except Exception:
        pass

    with torch.no_grad():
        pdb_str = model.infer_pdb(seq)
    with open(out_path, "w") as f:
        f.write(pdb_str)
    return out_path
# ---------------------------------------------------------
# FUNCIONES AUXILIARES PARA PDB
# ---------------------------------------------------------
def leer_archivo_pdb(pdb_path: str) -> str:
    """
    Lee un archivo PDB desde disco y devuelve su contenido como string.
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"No se encontró el archivo PDB: {pdb_path}")

    with open(pdb_path, "r") as f:
        return f.read()


def crear_estadisticas_detalladas(
    solutions,
    inicio_evaluaciones: int,
    max_evaluations: int,
    population_size: int
):
    """
    Devuelve un diccionario con estadísticas básicas del frente de Pareto.
    """
    stats = {
        "num_solutions": len(solutions),
        "population_size": population_size,
        "max_evaluations": max_evaluations,
    }

    if not solutions:
        return stats

    def collect_attr(key):
        vals = []
        for s in solutions:
            attrs = getattr(s, "attributes", {}) or {}
            v = attrs.get(key, None)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    for nombre, key in [
        ("rmsd", "rmsd"),
        ("gdt", "gdt"),
        ("design_energy", "design_energy"),
        ("tms_score", "tms_score"),
        ("fitness_total", "fitness_total"),
    ]:
        vals = collect_attr(key)
        if vals:
            stats[f"{nombre}_min"] = float(np.min(vals))
            stats[f"{nombre}_max"] = float(np.max(vals))
            stats[f"{nombre}_mean"] = float(np.mean(vals))
        else:
            stats[f"{nombre}_min"] = None
            stats[f"{nombre}_max"] = None
            stats[f"{nombre}_mean"] = None

    return stats


def mostrar_top_soluciones(solutions, top_n=5):
    """
    Muestra información de las mejores soluciones encontradas.
    """
    if not solutions:
        print("⚠️ No hay soluciones para mostrar.")
        return

    top_n = min(top_n, len(solutions))

    print(f"{'#':<3} {'RMSD':<8} {'GDT':<8} {'Energía':<10} {'TM-Score':<10} {'Fitness':<10}")
    print("-" * 60)

    for i, solution in enumerate(solutions[:top_n]):
        attrs = getattr(solution, "attributes", {}) or {}

        rmsd = attrs.get('rmsd', 'N/A')
        gdt = attrs.get('gdt', 'N/A')
        energy = attrs.get('design_energy', 'N/A')
        tms = attrs.get('tms_score', 'N/A')
        fitness = attrs.get('fitness_total', 'N/A')

        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)

        print(f"{i+1:<3} {fmt(rmsd):<8} {fmt(gdt):<8} {fmt(energy):<10} {fmt(tms):<10} {fmt(fitness):<10}")


def exportar_pareto_y_variables_csv(solutions, output_dir, file_name="PARETO_DETALLADO.csv"):
    """
    Exporta las soluciones no dominadas a un CSV con
    objetivos + algunos atributos útiles.
    """
    if not solutions:
        return

    os.makedirs(output_dir, exist_ok=True)
    ruta = os.path.join(output_dir, file_name)

    campos_base = ["index"]
    # f0, f1, f2...
    max_objs = max(len(sol.objectives) for sol in solutions)
    campos_obj = [f"f{i}" for i in range(max_objs)]

    # algunos atributos típicos:
    campos_attr = [
        "sequence",
        "amino_sequence",
        "rmsd",
        "gdt",
        "design_energy",
        "tms_score",
        "fitness_total",
        "kl_divergence",
    ]

    fieldnames = campos_base + campos_obj + campos_attr

    with open(ruta, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, sol in enumerate(solutions):
            row = {"index": idx}
            # objetivos
            for i, val in enumerate(sol.objectives):
                row[f"f{i}"] = val
            # atributos
            attrs = getattr(sol, "attributes", {}) or {}
            for k in campos_attr:
                if k in attrs:
                    row[k] = attrs[k]
            writer.writerow(row)

    print(f"CSV de Pareto detallado guardado en: {ruta}")

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, necesario para proyección 3d

def plot_pareto_front_3d(solutions, output_dir, file_name="PARETO_3D.png"):
    """
    Grafica el frente de Pareto (3 objetivos) en 3D.
    Si no hay 3 objetivos, sale silenciosamente.
    """
    if not solutions:
        return

    num_objs = len(solutions[0].objectives)
    if num_objs < 3:
        print("⚠️ Menos de 3 objetivos, no se genera gráfico 3D.")
        return

    xs = [s.objectives[0] for s in solutions]
    ys = [s.objectives[1] for s in solutions]
    zs = [s.objectives[2] for s in solutions]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs)

    ax.set_xlabel("f0 (RMSD)")
    ax.set_ylabel("f1 (-GDT)")
    ax.set_zlabel("f2 (Energy)")

    os.makedirs(output_dir, exist_ok=True)
    ruta = os.path.join(output_dir, file_name)
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)

    print(f"Gráfico 3D del frente de Pareto guardado en: {ruta}")


def elegir_mejor_solucion(solutions, prefer='rmsd'):
    """
    Elige una solución de la frontera de Pareto según un criterio:
    - 'rmsd': minimiza RMSD
    - 'gdt': maximiza GDT
    - 'energy' / 'energia': minimiza energía de diseño
    - 'fitness': minimiza fitness_total (si lo tratás como coste)
    """
    if not solutions:
        return None

    def get_attr(sol, key, default):
        attrs = getattr(sol, "attributes", {}) or {}
        return attrs.get(key, default)

    if prefer == 'gdt':
        # GDT es métrica de calidad, cuanto más grande mejor
        return max(solutions, key=lambda s: get_attr(s, 'gdt', float("-inf")))
    elif prefer in ('energy', 'energia'):
        return min(solutions, key=lambda s: get_attr(s, 'design_energy', float("inf")))
    elif prefer == 'fitness':
        return min(solutions, key=lambda s: get_attr(s, 'fitness_total', float("inf")))
    else:  # 'rmsd' por defecto
        return min(solutions, key=lambda s: get_attr(s, 'rmsd', float("inf")))
