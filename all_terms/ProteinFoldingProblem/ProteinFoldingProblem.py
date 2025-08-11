import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from typing import List
from transformers import EsmTokenizer, EsmModel
from ..Fitness.fitness import descriptores,mapa_contacto,fitness_gdt_rmsd_mc_fisquim,fitness_gdt_rmsd_mc,agrega_rmsd_gdt_E_MC_divKl,agrega_rmsd_gdt_E_MC
from ..pdb_seq_tools.pdb_seq_tools import extract_amino_acid_sequence_pdb,extract_backbone_atoms,det_sec,extract_backbone_atoms_str,extract_amino_acid_sequence
from Bio.PDB import PDBParser

class ProteinFoldingProblem(FloatProblem):
    """
    Problema de optimización multi-objetivo para el plegamiento de proteínas
    usando NSGA-II con métricas RMSD, GDT, mapas de contacto y descriptores fisicoquímicos.
    """
    
    def __init__(self, 
                 pdb_reference: str,
                 num_residues: int,
                 use_physicochemical_descriptors: bool = True,
                 corte: List[float] = [1.0, 2.0, 4.0, 8.0],
                 energia_params: tuple = (-50.0, 10.0)):
        """
        Inicializa el problema de plegamiento de proteínas.
        """
        
        # Parámetros del problema
        self.pdb_reference = pdb_reference
        self.num_residues = num_residues
        self.use_physicochemical_descriptors = use_physicochemical_descriptors
        self.corte = corte
        self.energia_a, self.energia_b = energia_params
        
        # Configuración del problema
        self._number_of_variables = num_residues * 4 * 3  # 4 átomos × 3 coordenadas
        self._number_of_objectives = 3  # Tres objetivos separados
        self._number_of_constraints = 0  # Sin restricciones
        
        # Definir límites para las coordenadas
        coord_min, coord_max = -50.0, 50.0
        self._lower_bound = [coord_min] * self._number_of_variables
        self._upper_bound = [coord_max] * self._number_of_variables
        
        # NO llamar al constructor padre aquí - jMetal maneja esto internamente
        
        # Extraer datos de referencia del PDB
        self._extract_reference_data()
        
        # Inicializar modelo ESM2 si se usan descriptores fisicoquímicos
        if self.use_physicochemical_descriptors:
            self._initialize_esm2_model()
    
    @property
    def number_of_variables(self) -> int:
        """Número de variables del problema."""
        return self._number_of_variables
    
    @property
    def number_of_objectives(self) -> int:
        """Número de objetivos del problema."""
        return self._number_of_objectives
    
    @property
    def number_of_constraints(self) -> int:
        """Número de restricciones del problema."""
        return self._number_of_constraints
    
    @property
    def lower_bound(self) -> List[float]:
        """Límites inferiores de las variables."""
        return self._lower_bound
    
    @property
    def upper_bound(self) -> List[float]:
        """Límites superiores de las variables."""
        return self._upper_bound
    
    def name(self) -> str:
        """Nombre del problema."""
        return "Protein Folding Optimization Problem"
    
    def _extract_reference_data(self):
        """Extrae los datos de referencia del PDB."""
        try:
            # Extraer backbone de referencia
            self.backbone_reference = extract_backbone_atoms_str(self.pdb_reference)
            
            # Extraer secuencia de aminoácidos
            self.amino_sequence = extract_amino_acid_sequence(self.pdb_reference)
            
            # Calcular mapa de contacto de referencia
            self.MC_reference = mapa_contacto(self.backbone_reference)
            
            print(f"Datos de referencia extraídos:")
            print(f"- Número de átomos del backbone: {len(self.backbone_reference)}")
            print(f"- Secuencia: {self.amino_sequence}")
            print(f"- Longitud de la secuencia: {len(self.amino_sequence)}")
            
        except Exception as e:
            raise ValueError(f"Error al extraer datos de referencia del PDB: {e}")
    
    def _initialize_esm2_model(self):
        """Inicializa el modelo ESM2 para descriptores fisicoquímicos."""
        try:
            self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.model_ESM2 = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.model_ESM2.eval()  # Modo evaluación
            
            # Calcular descriptores de referencia
            self.descriptor_ref = descriptores(
                self.amino_sequence, 
                self.tokenizer, 
                self.model_ESM2
            )
            
            print("Modelo ESM2 inicializado correctamente")
            
        except Exception as e:
            print(f"Advertencia: Error al inicializar ESM2, continuando sin descriptores: {e}")
            self.use_physicochemical_descriptors = False
    
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        """
        Evalúa una solución (configuración 3D de la proteína).
        """
        try:
            # Convertir variables de la solución a coordenadas 3D
            coords_3d = np.array(solution.variables).reshape(-1, 3)
            
            # Simular energía de diseño
            energia_design = self._calculate_design_energy(coords_3d)
            
            if self.use_physicochemical_descriptors:
                # Calcular descriptores de la configuración actual
                descriptor_temp = descriptores(
                    self.amino_sequence,
                    self.tokenizer,
                    self.model_ESM2
                )
                
                # Calcular métricas con descriptores fisicoquímicos
                rms, gdt, MC_similitud, divKl, distancias_sal, tms = fitness_gdt_rmsd_mc_fisquim(
                    self.backbone_reference,
                    coords_3d,
                    self.MC_reference,
                    self.corte,
                    self.descriptor_ref,
                    descriptor_temp
                )
                
                # Calcular fitness total
                fitness_total = agrega_rmsd_gdt_E_MC_divKl(
                    MC_similitud,
                    energia_design,
                    rms,
                    gdt,
                    divKl,
                    self.energia_a,
                    self.energia_b,
                    tms
                )
                
            else:
                # Calcular métricas sin descriptores fisicoquímicos
                rms, gdt, MC_similitud, distancias_sal, tms = fitness_gdt_rmsd_mc(
                    self.backbone_reference,
                    coords_3d,
                    self.MC_reference,
                    self.corte
                )
                
                # Calcular fitness total
                fitness_total = agrega_rmsd_gdt_E_MC(
                    MC_similitud,
                    energia_design,
                    rms,
                    gdt,
                    self.energia_a,
                    self.energia_b,
                    tms
                )
            
            # NSGA-II minimiza, así que configuramos los 3 objetivos
            # Objetivo 1: Minimizar RMSD
            solution.objectives[0] = rms
            
            # Objetivo 2: Maximizar GDT (convertimos a minimización)
            solution.objectives[1] = -gdt
            
            # Objetivo 3: Minimizar energía de diseño
            solution.objectives[2] = energia_design
            
            # Guardar métricas adicionales
            solution.attributes = {
                'rmsd': rms,
                'gdt': gdt,
                'mc_similarity': MC_similitud,
                'tms_score': tms,
                'design_energy': energia_design,
                'fitness_total': fitness_total
            }
            
            if self.use_physicochemical_descriptors:
                solution.attributes['kl_divergence'] = divKl
            
        except Exception as e:
            # En caso de error, asignar valores muy malos para los 3 objetivos
            print(f"Error en evaluación: {e}")
            solution.objectives[0] = float('inf')  # RMSD malo
            solution.objectives[1] = float('inf')  # GDT malo (ya negativizado)
            solution.objectives[2] = float('inf')  # Energía mala
            solution.attributes = {'error': str(e)}
        
        return solution
    
    def _calculate_design_energy(self, coords_3d: np.ndarray) -> float:
        """Calcula una energía de diseño simplificada."""
        # Ejemplo simple: energía basada en distancias entre átomos consecutivos
        distancias = np.linalg.norm(coords_3d[1:] - coords_3d[:-1], axis=1)
        
        # Penalizar distancias muy largas o muy cortas
        energia = np.sum(np.where(distancias < 1.0, (1.0 - distancias)**2, 0)) + \
                 np.sum(np.where(distancias > 5.0, (distancias - 5.0)**2, 0))
        
        return energia
    
    def create_solution(self) -> FloatSolution:
        """Crea una nueva solución con valores iniciales aleatorios."""
        new_solution = FloatSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives
        )
        
        # Inicializar con valores aleatorios dentro de los límites
        new_solution.variables = [
            np.random.uniform(self.lower_bound[i], self.upper_bound[i])
            for i in range(self.number_of_variables)
        ]
        
        return new_solution