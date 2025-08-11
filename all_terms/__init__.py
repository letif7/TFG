from .Algorithm_evolutionary.algorithm_evolutionary import EDA_isla
import os
import pyrosetta

def run(max_generations = 1000, 
        population_size = 5, 
        sample_size = 3,
        input_folder = 'target_pdbs'):
    pyrosetta.init()
    directorio_trabajo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_carpeta_bd_selected = os.path.join(directorio_trabajo, input_folder)
    archivos_en_bd_selected = os.listdir(ruta_carpeta_bd_selected)
    for i in archivos_en_bd_selected:
        pdb_file_path = os.path.join(ruta_carpeta_bd_selected,i)
        print(pdb_file_path)

        "se determinaran los descriptores fisicoquimicos 1 si si 0 si no"
        desc_si_no=1

        "ESMfold se ejecuta de manera local"
        local=0

        "Cuando se esta trabajando en colab"
        nueva_ruta='algo'

        "1 si es continuacion de una ejecucion anterior, 0 otro caso"
        # if i=='cesarp.pdb':
        #     continua=1
        # else:
        #     continua=0
        continua = 0

        "ultimo dato guardado de las ejecuciones anteriores, rellenar si continua=1"
        numnum=460
        "valores de a y b en la funcion objetivo para dilatar la energia"
        a=30
        b=30

        "cantidad de individuos por los que estara formada la poblacion"
        n_pop=population_size
        "ejecuciones de cada individuo por etapa"
        n_rep=sample_size
        "Cantidad de ejecuciones a determinar el fitnes"
        n_fit=3
        "cantidad a actualizar del propio"
        act_prop=2
        "cantidad a actualizar del global"
        act_glob=1
        "cantidad de mejores elementos guardados (las ejecuciones de menor fitness)"
        n_best=80
        "posicion en el arreglo donde se encuetra el elmento de menor fitness"
        pos_min=n_best-1
        "Numero de generaciones"
        n_generaciones=max_generations
        "cantidad de elementod de la poblacion que se reiniciaran"
        n_reinicia=8
        "numero de generaciones para reiniciarce"
        #n_genra_reinicio=50
        n_genra_reinicio=20

        "Corte para el GDT"
        corte=[1,2,4,8]

        "Variables y nombres"
        aminoacidos_nombre = ['Valina','Leucina','Isoleucina','Metionina','Fenilalanina','Lisina','Arginina','Histidina','Ácido aspártico','Ácido glutámico','Asparagina','Glutamina','Tirosina','Triptófano','Serina','Treonina','Prolina','Alanina','Glicina','Cisteína']
        amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        "Islas a utilizar en el algoritmo"
        islas=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        #islas=[15,16,17,18,19,20]

        "con que probabilidad va ha haber coperacion entre las islas"
        prob_copera=0.10

        "cantidad de elementos a ser seleccionados para colaborar"
        n_colab=1
        #EDA_tres_capas(pdb_file_path,desc_si_no,continua,numnum,n_pop,n_rep,n_fit,act_prop,act_glob,n_best,pos_min,n_generaciones,n_reinicia,n_genra_reinicio,corte,amino,a,b,local,nueva_ruta)
        #poblacion, best_execution,best_fitness,pdb_select,best_fitness_energ=Algoritmo_evolutivo_energia_MC_descriptor(continua,n,n_pop,n_rep,n_fit,act_prop,act_glob,amino,num_amino,n_best,n_generaciones,pos_min,BB,w1,w2,corte,tokenizer,model,nrep,mat_frec,nombre_sin_extension,n_func,MC_BB,ruta_archivo,numnum,secuencia_ref)
        "cantidad de elementos utilizados para actualizar la red"
        tot_act_red=1500

        "detener cuando se tenga un gdt deceado"
        max_gdt=1.2

        EDA_isla(pdb_file_path,desc_si_no,continua,numnum,n_pop,n_rep,n_fit,act_prop,act_glob,n_best,pos_min,n_generaciones,
                    n_reinicia,n_genra_reinicio,corte,amino,a,b,local,nueva_ruta, islas,prob_copera,n_colab,tot_act_red,max_gdt)


# __init__.py
import numpy as np
from Bio.PDB import PDBParser
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file
from typing import List, Optional, Tuple
import os
import datetime

# Importar la clase del problema
from .ProteinFoldingProblem.ProteinFoldingProblem import ProteinFoldingProblem


def contar_residuos(pdb_file: str) -> int:
    """
    Cuenta el número de residuos únicos en un archivo PDB.
    
    Args:
        pdb_file: Ruta al archivo PDB
        
    Returns:
        int: Número de residuos únicos en la proteína
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    residuos_unicos = set()
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":  # Solo aminoácidos estándar
                    residuos_unicos.add((chain.id, residue.id[1]))
    
    return len(residuos_unicos)


def leer_archivo_pdb(pdb_file: str) -> str:
    """
    Lee el contenido de un archivo PDB y lo retorna como string.
    
    Args:
        pdb_file: Ruta al archivo PDB
        
    Returns:
        str: Contenido del archivo PDB
    """
    try:
        with open(pdb_file, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo PDB: {pdb_file}")
    except Exception as e:
        raise Exception(f"Error al leer el archivo PDB: {e}")


def obtener_primer_pdb(input_folder='demo_pdbs'):
    """
    Obtiene la ruta del primer archivo en la carpeta especificada.
    
    Args:
        input_folder (str): Nombre de la carpeta que contiene los archivos PDB
        
    Returns:
        str: Ruta completa del primer archivo encontrado
        
    Raises:
        FileNotFoundError: Si la carpeta no existe
        ValueError: Si no hay archivos en la carpeta
    """
    # Obtener directorio de trabajo (2 niveles arriba del archivo actual)
    directorio_trabajo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construir ruta a la carpeta de PDBs
    ruta_carpeta_bd_selected = os.path.join(directorio_trabajo, input_folder)
    
    # Verificar que la carpeta existe
    if not os.path.exists(ruta_carpeta_bd_selected):
        raise FileNotFoundError(f"La carpeta {ruta_carpeta_bd_selected} no existe")
    
    # Listar archivos en la carpeta
    archivos_en_bd_selected = os.listdir(ruta_carpeta_bd_selected)
    
    # Verificar que hay archivos
    if not archivos_en_bd_selected:
        raise ValueError(f"No hay archivos en la carpeta {ruta_carpeta_bd_selected}")
    
    # Obtener el primer archivo y construir ruta completa
    primer_archivo = archivos_en_bd_selected[0]
    pdb_file_path = os.path.join(ruta_carpeta_bd_selected, primer_archivo)
    
    return pdb_file_path


def obtener_soluciones_del_algoritmo(algorithm):
    """
    Obtiene las soluciones del algoritmo NSGA-II de manera robusta,
    compatible con diferentes versiones de jMetal.
    
    Args:
        algorithm: Instancia del algoritmo NSGA-II ejecutado
        
    Returns:
        List: Lista de soluciones encontradas por el algoritmo
    """
    # Intentar diferentes métodos según la versión de jMetal
    methods_to_try = [
        'get_result',      # Método más común en versiones recientes
        'get_results',     # Variante con 's'
        'result',          # Propiedad directa
        'solutions',       # Otra propiedad posible
        'population'       # En algunas versiones
    ]
    
    for method_name in methods_to_try:
        try:
            if hasattr(algorithm, method_name):
                method_or_attr = getattr(algorithm, method_name)
                
                # Si es un método (callable), llamarlo
                if callable(method_or_attr):
                    return method_or_attr()
                # Si es una propiedad, accederla directamente
                else:
                    return method_or_attr
        except Exception as e:
            print(f"⚠️ Método {method_name} falló: {e}")
            continue
    
    # Si ningún método funciona, intentar acceso directo a atributos internos
    possible_attributes = ['_population', '_result', '_solutions']
    for attr_name in possible_attributes:
        if hasattr(algorithm, attr_name):
            try:
                return getattr(algorithm, attr_name)
            except Exception:
                continue
    
    raise AttributeError(
        f"No se pudo obtener las soluciones del algoritmo. "
        f"Métodos intentados: {methods_to_try}. "
        f"Atributos disponibles: {[attr for attr in dir(algorithm) if not attr.startswith('_')]}"
    )


def optimizar_plegamiento_proteina(
    pdb_reference_file: str = None,
    max_evaluations: int = 2500,
    population_size: int = 100,
    use_physicochemical_descriptors: bool = True,
    corte: List[float] = [1.0, 2.0, 4.0, 8.0],
    energia_params: Tuple[float, float] = (-50.0, 10.0),
    crossover_probability: float = 0.9,
    mutation_probability: float = 0.1,
    crossover_distribution_index: float = 20.0,
    mutation_distribution_index: float = 20.0,
    output_dir: str = "results",
    verbose: bool = True
) -> Tuple[List, dict]:
    """
    Optimiza el plegamiento de una proteína usando NSGA-II.
    Automáticamente cuenta los residuos del archivo PDB y crea el problema.
    
    Args:
        pdb_reference_file: Ruta al archivo PDB de referencia (por defecto usa 1y32.pdb)
        max_evaluations: Número máximo de evaluaciones
        population_size: Tamaño de la población
        use_physicochemical_descriptors: Si usar descriptores fisicoquímicos
        corte: Lista de distancias de corte para GDT
        energia_params: Parámetros (a, b) para la función de energía
        crossover_probability: Probabilidad de cruzamiento
        mutation_probability: Probabilidad de mutación
        crossover_distribution_index: Índice de distribución para cruzamiento SBX
        mutation_distribution_index: Índice de distribución para mutación polinomial
        output_dir: Directorio para guardar resultados
        verbose: Si mostrar información durante la ejecución
        
    Returns:
        Tuple[List, dict]: Tupla con (soluciones no dominadas, estadísticas del proceso)
    """
    
    if pdb_reference_file is None:
        pdb_reference_file = obtener_primer_pdb()

    if verbose:
        print("="*60)
        print("OPTIMIZACIÓN DE PLEGAMIENTO DE PROTEÍNAS CON NSGA-II")
        print("="*60)
        print(f"Archivo PDB de referencia: {pdb_reference_file}")
    
    # Verificar que el archivo existe
    if not os.path.exists(pdb_reference_file):
        raise FileNotFoundError(f"No se encontró el archivo PDB: {pdb_reference_file}")
    
    # Contar residuos automáticamente
    if verbose:
        print("Contando residuos en el archivo PDB...")
    
    num_residues = contar_residuos(pdb_reference_file)
    
    # Leer el contenido del PDB como string
    if verbose:
        print("Leyendo contenido del archivo PDB...")
    
    pdb_content = leer_archivo_pdb(pdb_reference_file)
    
    if verbose:
        print(f"✓ Número de residuos detectados: {num_residues}")
        print(f"✓ Contenido PDB leído: {len(pdb_content)} caracteres")
        print(f"✓ Usar descriptores fisicoquímicos: {use_physicochemical_descriptors}")
        print(f"✓ Parámetros de energía: a={energia_params[0]}, b={energia_params[1]}")
    
    # Crear el problema con los parámetros calculados automáticamente
    try:
        if verbose:
            print("\nCreando problema de plegamiento de proteínas...")
            
        problem = ProteinFoldingProblem(
            pdb_reference=pdb_content,  # Contenido del PDB como string
            num_residues=num_residues,  # Número de residuos calculado automáticamente
            use_physicochemical_descriptors=use_physicochemical_descriptors,
            corte=corte,
            energia_params=energia_params
        )
        
        if verbose:
            print(f"✓ Problema creado exitosamente:")
            print(f"  - Variables: {problem.number_of_variables}")
            print(f"  - Objetivos: {problem.number_of_objectives}")
            print(f"  - Residuos: {problem.num_residues}")
            
    except Exception as e:
        # DIAGNÓSTICO MEJORADO: Mostrar más detalles del error
        import traceback
        print(f"\nERROR DETALLADO:")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        print(f"\nTraceback completo:")
        traceback.print_exc()
        
        # Verificar las importaciones
        print(f"\nDIAGNÓSTICO DE IMPORTACIONES:")
        try:
            print(f"ProteinFoldingProblem type: {type(ProteinFoldingProblem)}")
            print(f"ProteinFoldingProblem: {ProteinFoldingProblem}")
        except NameError:
            print("ERROR: ProteinFoldingProblem no está definido")
        
        raise Exception(f"Error al crear el problema de plegamiento: {e}")
    
    # Configurar operadores genéticos
    crossover = SBXCrossover(
        probability=crossover_probability, 
        distribution_index=crossover_distribution_index
    )
    
    mutation = PolynomialMutation(
        probability=mutation_probability / problem.number_of_variables,
        distribution_index=mutation_distribution_index
    )
    
    # Crear el algoritmo NSGA-II
    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations)
    )
    
    if verbose:
        print("\nConfiguración del algoritmo:")
        print(f"- Algoritmo: NSGA-II")
        print(f"- Tamaño de población: {population_size}")
        print(f"- Máximo de evaluaciones: {max_evaluations}")
        print(f"- Probabilidad de cruzamiento: {crossover_probability}")
        print(f"- Probabilidad de mutación: {mutation_probability}")
        print("\nIniciando optimización...")
    
    # Ejecutar el algoritmo
    start_time = datetime.datetime.now()
    algorithm.run()
    end_time = datetime.datetime.now()
    
    # Obtener resultados de manera robusta
    try:
        if verbose:
            print("Obteniendo resultados...")
        
        solutions = obtener_soluciones_del_algoritmo(algorithm)
        
        if verbose:
            print(f"✓ Soluciones obtenidas: {len(solutions)}")
            
    except Exception as e:
        print(f"❌ Error al obtener soluciones: {e}")
        print("Información de depuración del algoritmo:")
        print(f"Tipo de algoritmo: {type(algorithm)}")
        print(f"Atributos disponibles: {[attr for attr in dir(algorithm) if not attr.startswith('__')]}")
        raise
    
    # Obtener soluciones no dominadas
    non_dominated_solutions = get_non_dominated_solutions(solutions)
    
    execution_time = (end_time - start_time).total_seconds()
    
    if verbose:
        print(f"\n✓ Optimización completada!")
        print(f"⏱️ Tiempo de ejecución: {execution_time:.2f} segundos")
        print(f"🧬 Soluciones totales: {len(solutions)}")
        print(f"🏆 Soluciones no dominadas: {len(non_dominated_solutions)}")
    
    # Crear directorio de resultados si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Guardar resultados
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar valores de función objetivo
    objectives_file = os.path.join(output_dir, f"objectives_{timestamp}.txt")
    print_function_values_to_file(non_dominated_solutions, objectives_file)
    
    # Guardar variables (coordenadas)
    variables_file = os.path.join(output_dir, f"variables_{timestamp}.txt")
    print_variables_to_file(non_dominated_solutions, variables_file)
    
    # Crear estadísticas detalladas
    stats = crear_estadisticas_detalladas(non_dominated_solutions, execution_time, 
                                        max_evaluations, population_size)
    
    # Guardar estadísticas
    stats_file = os.path.join(output_dir, f"statistics_{timestamp}.txt")
    guardar_estadisticas(stats, stats_file)
    
    if verbose:
        print(f"\n💾 Resultados guardados en:")
        print(f"  - Objetivos: {objectives_file}")
        print(f"  - Variables: {variables_file}")
        print(f"  - Estadísticas: {stats_file}")
        
        # Mostrar estadísticas básicas
        if non_dominated_solutions:
            print(f"\n📊 Análisis de las mejores soluciones:")
            mostrar_top_soluciones(non_dominated_solutions, top_n=5)
    
    return non_dominated_solutions, stats


def obtener_soluciones_del_algoritmo(algorithm):
    """
    Obtiene las soluciones del algoritmo NSGA-II de manera robusta,
    compatible con diferentes versiones de jMetal.
    
    Args:
        algorithm: Instancia del algoritmo NSGA-II ejecutado
        
    Returns:
        List: Lista de soluciones encontradas por el algoritmo
    """
    # Intentar diferentes métodos según la versión de jMetal
    methods_to_try = [
        'get_result',      # Método más común en versiones recientes
        'get_results',     # Variante con 's'
        'result',          # Propiedad directa
        'solutions',       # Otra propiedad posible
        'population'       # En algunas versiones
    ]
    
    for method_name in methods_to_try:
        try:
            if hasattr(algorithm, method_name):
                method_or_attr = getattr(algorithm, method_name)
                
                # Si es un método (callable), llamarlo
                if callable(method_or_attr):
                    result = method_or_attr()
                    if result is not None:
                        return result
                # Si es una propiedad, accederla directamente
                else:
                    if method_or_attr is not None:
                        return method_or_attr
        except Exception as e:
            print(f"⚠️ Método {method_name} falló: {e}")
            continue
    
    # Si ningún método funciona, intentar acceso directo a atributos internos
    possible_attributes = ['_population', '_result', '_solutions', 'population_']
    for attr_name in possible_attributes:
        if hasattr(algorithm, attr_name):
            try:
                attr_value = getattr(algorithm, attr_name)
                if attr_value is not None:
                    return attr_value
            except Exception:
                continue
    
    raise AttributeError(
        f"No se pudo obtener las soluciones del algoritmo. "
        f"Métodos intentados: {methods_to_try}. "
        f"Atributos disponibles: {[attr for attr in dir(algorithm) if not attr.startswith('__')]}"
    )


def mostrar_top_soluciones(solutions, top_n=5):
    """
    Muestra información de las mejores soluciones encontradas.
    
    Args:
        solutions: Lista de soluciones no dominadas
        top_n: Número de soluciones a mostrar
    """
    print(f"{'#':<3} {'RMSD':<8} {'GDT':<8} {'Energía':<10} {'TM-Score':<10} {'Fitness':<10}")
    print("-" * 60)
    
    for i, solution in enumerate(solutions[:top_n]):
        if hasattr(solution, 'attributes') and solution.attributes:
            attrs = solution.attributes
            
            # Extraer valores con valores por defecto
            rmsd = attrs.get('rmsd', 'N/A')
            gdt = attrs.get('gdt', 'N/A')
            energy = attrs.get('design_energy', 'N/A')
            tms = attrs.get('tms_score', 'N/A')
            fitness = attrs.get('fitness_total', 'N/A')
            
            # Formatear números
            rmsd_str = f"{rmsd:.3f}" if isinstance(rmsd, (int, float)) else str(rmsd)
            gdt_str = f"{gdt:.3f}" if isinstance(gdt, (int, float)) else str(gdt)
            energy_str = f"{energy:.3f}" if isinstance(energy, (int, float)) else str(energy)
            tms_str = f"{tms:.3f}" if isinstance(tms, (int, float)) else str(tms)
            fitness_str = f"{fitness:.3f}" if isinstance(fitness, (int, float)) else str(fitness)
            
            print(f"{i+1:<3} {rmsd_str:<8} {gdt_str:<8} {energy_str:<10} {tms_str:<10} {fitness_str:<10}")
        else:
            # Si no hay atributos, mostrar solo los objetivos
            obj_str = " ".join([f"{obj:.3f}" for obj in solution.objectives])
            print(f"{i+1:<3} Objetivos: {obj_str}")


def crear_estadisticas_detalladas(solutions: List, execution_time: float, 
                                max_evaluations: int, population_size: int) -> dict:
    """
    Crea estadísticas detalladas de los resultados de la optimización.
    
    Args:
        solutions: Lista de soluciones no dominadas
        execution_time: Tiempo de ejecución en segundos
        max_evaluations: Número máximo de evaluaciones
        population_size: Tamaño de la población
        
    Returns:
        dict: Diccionario con estadísticas detalladas
    """
    stats = {
        'execution_time': execution_time,
        'max_evaluations': max_evaluations,
        'population_size': population_size,
        'num_solutions': len(solutions),
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    if solutions:
        # Extraer métricas de fitness de las soluciones
        fitness_values = []
        rmsd_values = []
        gdt_values = []
        mc_similarity_values = []
        tms_values = []
        energy_values = []
        
        for sol in solutions:
            if hasattr(sol, 'attributes') and sol.attributes:
                attrs = sol.attributes
                if 'fitness_total' in attrs:
                    fitness_values.append(attrs['fitness_total'])
                if 'rmsd' in attrs:
                    rmsd_values.append(attrs['rmsd'])
                if 'gdt' in attrs:
                    gdt_values.append(attrs['gdt'])
                if 'mc_similarity' in attrs:
                    mc_similarity_values.append(attrs['mc_similarity'])
                if 'tms_score' in attrs:
                    tms_values.append(attrs['tms_score'])
                if 'design_energy' in attrs:
                    energy_values.append(attrs['design_energy'])
        
        # Calcular estadísticas para cada métrica
        for metric_name, values in [
            ('fitness', fitness_values),
            ('rmsd', rmsd_values),
            ('gdt', gdt_values),
            ('mc_similarity', mc_similarity_values),
            ('tms_score', tms_values),
            ('design_energy', energy_values)
        ]:
            if values:
                stats[f'{metric_name}_min'] = float(np.min(values))
                stats[f'{metric_name}_max'] = float(np.max(values))
                stats[f'{metric_name}_mean'] = float(np.mean(values))
                stats[f'{metric_name}_std'] = float(np.std(values))
                stats[f'{metric_name}_median'] = float(np.median(values))
    
    return stats


def guardar_estadisticas(stats: dict, filename: str):
    """
    Guarda las estadísticas en un archivo de texto.
    
    Args:
        stats: Diccionario con estadísticas
        filename: Nombre del archivo donde guardar
    """
    with open(filename, 'w') as f:
        f.write("ESTADÍSTICAS DE OPTIMIZACIÓN DE PLEGAMIENTO DE PROTEÍNAS\n")
        f.write("="*60 + "\n\n")
        
        f.write("CONFIGURACIÓN:\n")
        f.write(f"Tiempo de ejecución: {stats['execution_time']:.2f} segundos\n")
        f.write(f"Máximo de evaluaciones: {stats['max_evaluations']}\n")
        f.write(f"Tamaño de población: {stats['population_size']}\n")
        f.write(f"Soluciones no dominadas: {stats['num_solutions']}\n")
        f.write(f"Timestamp: {stats['timestamp']}\n\n")
        
        # Escribir estadísticas de métricas si están disponibles
        metrics = ['fitness', 'rmsd', 'gdt', 'mc_similarity', 'tms_score', 'design_energy']
        for metric in metrics:
            if f'{metric}_min' in stats:
                f.write(f"{metric.upper().replace('_', ' ')} ESTADÍSTICAS:\n")
                f.write(f"  Mínimo: {stats[f'{metric}_min']:.6f}\n")
                f.write(f"  Máximo: {stats[f'{metric}_max']:.6f}\n")
                f.write(f"  Media: {stats[f'{metric}_mean']:.6f}\n")
                f.write(f"  Desviación estándar: {stats[f'{metric}_std']:.6f}\n")
                f.write(f"  Mediana: {stats[f'{metric}_median']:.6f}\n\n")


def ejemplo_uso():
    """
    Función de ejemplo que muestra cómo usar la optimización.
    """
    print("Ejemplo de uso de la optimización de plegamiento de proteínas:")
    print("\n# Uso básico:")
    print("solutions, stats = optimizar_plegamiento_proteina('protein.pdb')")
    
    print("\n# Uso con parámetros personalizados:")
    print("""solutions, stats = optimizar_plegamiento_proteina(
    pdb_reference_file='protein.pdb',
    max_evaluations=50000,
    population_size=200,
    use_physicochemical_descriptors=True,
    crossover_probability=0.95,
    mutation_probability=0.05,
    output_dir='my_results'
)""")
