import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# jMetal imports
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file

# BioPython
from Bio.PDB import PDBParser

# Importar tus clases locales (asegúrate de que estén en la misma carpeta de Colab)
from ProteinFoldingProblem import ProteinFoldingProblem
from pdb_seq_tools import extract_amino_acid_sequence, det_sec, extract_backbone_atoms_str
from Algorithm_evolutionary.algorithm_evolutionary import EDA_isla


def optimizar_plegamiento_proteina(
    pdb_reference_file: str = None,
    max_evaluations: int = 2500,
    population_size: int = 100,
    use_physicochemical_descriptors: bool = True,
    corte: list = [1.0, 2.0, 4.0, 8.0],
    energia_params: tuple = (-50.0, 10.0),
    crossover_probability: float = 0.9,
    mutation_probability: float = 0.1,
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

    # Operadores genéticos
    crossover = SBXCrossover(probability=crossover_probability, distribution_index=crossover_distribution_index)
    mutation = PolynomialMutation(probability=mutation_probability / problem.number_of_variables,
                                 distribution_index=mutation_distribution_index)

    # Algoritmo NSGA-II
    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations)
    )

    algorithm.run()

    solutions = obtener_soluciones_del_algoritmo(algorithm)
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


# Funciones auxiliares: guardar PDB, leer archivos, contar residuos, etc.
# (copiar todas tus funciones de antes, pero eliminar PyRosetta)

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
