import os

def obtener_primer_pdb(input_folder='target_pdbs'):
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


def obtener_todos_los_pdb(input_folder='target_pdbs'):
    """
    Obtiene las rutas de todos los archivos en la carpeta especificada.
    
    Args:
        input_folder (str): Nombre de la carpeta que contiene los archivos PDB
        
    Returns:
        list: Lista con las rutas completas de todos los archivos
    """
    # Obtener directorio de trabajo
    directorio_trabajo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construir ruta a la carpeta
    ruta_carpeta_bd_selected = os.path.join(directorio_trabajo, input_folder)
    
    # Verificar que la carpeta existe
    if not os.path.exists(ruta_carpeta_bd_selected):
        raise FileNotFoundError(f"La carpeta {ruta_carpeta_bd_selected} no existe")
    
    # Listar archivos y construir rutas completas
    archivos_en_bd_selected = os.listdir(ruta_carpeta_bd_selected)
    rutas_completas = []
    
    for archivo in archivos_en_bd_selected:
        pdb_file_path = os.path.join(ruta_carpeta_bd_selected, archivo)
        rutas_completas.append(pdb_file_path)
    
    return rutas_completas


def obtener_archivos_pdb_filtrados(input_folder='target_pdbs', extension='.pdb'):
    """
    Obtiene solo los archivos con extensión .pdb de la carpeta especificada.
    
    Args:
        input_folder (str): Nombre de la carpeta que contiene los archivos
        extension (str): Extensión de archivo a filtrar (por defecto '.pdb')
        
    Returns:
        list: Lista con las rutas completas de archivos PDB
    """
    # Obtener directorio de trabajo
    directorio_trabajo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construir ruta a la carpeta
    ruta_carpeta_bd_selected = os.path.join(directorio_trabajo, input_folder)
    
    # Verificar que la carpeta existe
    if not os.path.exists(ruta_carpeta_bd_selected):
        raise FileNotFoundError(f"La carpeta {ruta_carpeta_bd_selected} no existe")
    
    # Listar y filtrar archivos PDB
    archivos_en_bd_selected = os.listdir(ruta_carpeta_bd_selected)
    archivos_pdb = [f for f in archivos_en_bd_selected if f.endswith(extension)]
    
    # Construir rutas completas
    rutas_pdb = []
    for archivo in archivos_pdb:
        pdb_file_path = os.path.join(ruta_carpeta_bd_selected, archivo)
        rutas_pdb.append(pdb_file_path)
    
    return rutas_pdb


# Ejemplo de uso
if __name__ == "__main__":
    try:
        # Obtener primer archivo
        primer_pdb = obtener_primer_pdb()
        print(f"Primer archivo PDB: {primer_pdb}")
        
        # Obtener todos los archivos
        todos_los_archivos = obtener_todos_los_pdb()
        print(f"\nTodos los archivos ({len(todos_los_archivos)}):")
        for i, archivo in enumerate(todos_los_archivos, 1):
            print(f"{i}. {archivo}")
        
        # Obtener solo archivos .pdb
        solo_pdbs = obtener_archivos_pdb_filtrados()
        print(f"\nSolo archivos .pdb ({len(solo_pdbs)}):")
        for i, archivo in enumerate(solo_pdbs, 1):
            print(f"{i}. {archivo}")
            
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")