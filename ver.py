# QUITA EL PUNTO DEL IMPORT
from .all_terms import optimizar_plegamiento_proteina as run_nsga_ii

def main():
    print("Iniciando optimización de plegamiento de proteínas...")
    
    try:
        # Ejecutar la optimización
        solutions, stats = run_nsga_ii(
            max_evaluations=5000,   # Prueba rápida
            population_size=30,     # Población pequeña
            verbose=True           # Ver el progreso
        )
        
        print(f"\n¡Optimización completada!")
        print(f"Soluciones encontradas: {len(solutions)}")
        print(f"Tiempo de ejecución: {stats['execution_time']:.2f} segundos")
        
        # La mejor solución
        mejor = min(solutions, key=lambda s: s.objectives[0])
        if hasattr(mejor, 'attributes'):
            print(f"Mejor fitness: {mejor.attributes.get('fitness_total', 'N/A')}")
            
    except Exception as e:
        print(f"Error durante la optimización: {e}")

if __name__ == "__main__":
    main()
