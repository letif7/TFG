import os
from .all_terms import  optimizar_plegamiento_proteina

def main():
    """Función principal que ejecuta la optimización de plegamiento de proteínas."""
    try:
        print("="*60)
        print("INICIO DE LA OPTIMIZACIÓN DE PLEGAMIENTO DE PROTEÍNAS")
        print("="*60)

        # Carpeta base donde está este archivo (__main__.py)
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Ruta al PDB dentro de tu proyecto (Google Colab usa /content/TFG/...)
        pdb_file = os.path.join(base_dir, "demo_pdbs", "1y32.pdb")

        # Ejecutar la optimización con parámetros por defecto
        solutions, stats = optimizar_plegamiento_proteina(
            pdb_reference_file=pdb_file,
            max_evaluations=60,  # pruebas
            population_size=50,  # pruebas
            verbose=True
        )

        if solutions:
            print(f"\nOptimización completada exitosamente!")
            print(f"Se encontraron {len(solutions)} soluciones no dominadas")
            exec_time = stats.get("execution_time", None)

            if isinstance(exec_time, (int, float)):
                print(f"Tiempo de ejecución: {exec_time:.2f} segundos")
            else:
                print(f"Tiempo de ejecución: {exec_time or 'N/A'} segundos")

           
        else:
            print("La optimización se ejecutó, pero no se obtuvieron soluciones.")

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
        return 1
    except Exception as e:
        print(f"Error durante la optimización: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
