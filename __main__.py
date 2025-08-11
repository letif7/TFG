from .all_terms import optimizar_plegamiento_proteina

def main():
    """Función principal que ejecuta la optimización."""
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

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
