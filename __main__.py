"""from .all_terms import optimizar_plegamiento_proteina

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

if __name__ == "__main__":
    plot_latest_objectives()