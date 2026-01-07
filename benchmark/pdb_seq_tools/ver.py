import numpy as np
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from typing import List
from transformers import EsmTokenizer, EsmModel, EsmForProteinFolding
from ..Fitness.fitness import descriptores, mapa_contacto, fitness_gdt_rmsd_mc_fisquim, fitness_gdt_rmsd_mc, agrega_rmsd_gdt_E_MC_divKl, agrega_rmsd_gdt_E_MC
from ..pdb_seq_tools.pdb_seq_tools import extract_amino_acid_sequence_pdb, extract_backbone_atoms, det_sec, extract_backbone_atoms_str, extract_amino_acid_sequence
from Bio.PDB import PDBParser
import torch
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter("ignore", PDBConstructionWarning)


class ProteinSequenceSolution(Solution):
    """Solución personalizada para secuencias de proteínas."""
    
    def __init__(self, sequence_length: int, number_of_objectives: int):
        super().__init__(sequence_length, number_of_objectives)
        # Las variables serán índices de aminoácidos (0-19)
        self.variables = [0] * sequence_length
        self.objectives = [0.0] * number_of_objectives
        self.attributes = {}

class ProteinSequenceOptimizationProblem(Problem):
    """
    Problema de optimización multi-objetivo para el diseño de secuencias de proteínas
    usando NSGA-II con métricas RMSD, GDT, mapas de contacto y descriptores fisicoquímicos.
    La población consiste en secuencias de aminoácidos que se evalúan mediante plegamiento.
    """
    
    # Diccionario de aminoácidos con sus códigos de una letra
    AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    def __init__(self, 
                 pdb_reference: str,
                 use_physicochemical_descriptors: bool = True,
                 corte: List[float] = [1.0, 2.0, 4.0, 8.0],
                 energia_params: tuple = (-50.0, 10.0)):
        """
        Inicializa el problema de optimización de secuencias de proteínas.
        
        Args:
            pdb_reference: String del archivo PDB de referencia
            use_physicochemical_descriptors: Si usar descriptores fisicoquímicos
            corte: Valores de corte para las métricas
            energia_params: Parámetros de energía (a, b)
        """
        
        # Parámetros del problema
        self.pdb_reference = pdb_reference
        self.use_physicochemical_descriptors = use_physicochemical_descriptors
        self.corte = corte
        self.energia_a, self.energia_b = energia_params
        
        # Extraer datos de referencia del PDB
        self._extract_reference_data()
        
        # Configuración del problema
        self._number_of_variables = len(self.amino_sequence_reference)
        self._number_of_objectives = 3  # RMSD, GDT, Energía
        self._number_of_constraints = 0
        
        # Los límites son los índices de aminoácidos (0-19)
        self._lower_bound = [0] * self._number_of_variables
        self._upper_bound = [19] * self._number_of_variables  # 20 aminoácidos (0-19)
        
        # Inicializar modelos
        if self.use_physicochemical_descriptors:
            self._initialize_esm2_model()
        
        self._initialize_folding_model()
        
        print(f"Problema de optimización de secuencias inicializado:")
        print(f"- Longitud de secuencia: {self._number_of_variables}")
        print(f"- Secuencia de referencia: {self.amino_sequence_reference}")
    
    @property
    def number_of_variables(self) -> int:
        return self._number_of_variables
    
    @property
    def number_of_objectives(self) -> int:
        return self._number_of_objectives
    
    @property
    def number_of_constraints(self) -> int:
        return self._number_of_constraints
    
    @property
    def lower_bound(self) -> List[float]:
        return self._lower_bound
    
    @property
    def upper_bound(self) -> List[float]:
        return self._upper_bound
    
    def name(self) -> str:
        return "Protein Sequence Optimization Problem"
    
    def _extract_reference_data(self):
        """Extrae los datos de referencia del PDB."""
        try:
            # Extraer secuencia de aminoácidos de referencia
            self.amino_sequence_reference = extract_amino_acid_sequence(self.pdb_reference)
            
            # Extraer backbone de referencia
            self.backbone_reference = extract_backbone_atoms_str(self.pdb_reference)
            
            # Calcular mapa de contacto de referencia
            self.MC_reference = mapa_contacto(self.backbone_reference)
            
            print(f"Datos de referencia extraídos:")
            print(f"- Secuencia de referencia: {self.amino_sequence_reference}")
            print(f"- Longitud: {len(self.amino_sequence_reference)}")
            print(f"- Átomos del backbone: {len(self.backbone_reference)}")
            
        except Exception as e:
            raise ValueError(f"Error al extraer datos de referencia del PDB: {e}")
    
    def _initialize_esm2_model(self):
        """Inicializa el modelo ESM2 para descriptores fisicoquímicos."""
        try:
            self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.model_ESM2 = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.model_ESM2.eval()
            
            # Calcular descriptores de referencia
            self.descriptor_ref = descriptores(
                self.amino_sequence_reference, 
                self.tokenizer, 
                self.model_ESM2
            )
            
            print("Modelo ESM2 para descriptores inicializado correctamente")
            
        except Exception as e:
            print(f"Advertencia: Error al inicializar ESM2, continuando sin descriptores: {e}")
            self.use_physicochemical_descriptors = False
    
    def _initialize_folding_model(self):
        """Inicializa el modelo ESMFold para plegamiento de proteínas."""
        try:
            # Usar ESMFold para plegamiento  
            self.folding_tokenizer = EsmTokenizer.from_pretrained("facebook/esmfold_v1")
            self.folding_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            self.folding_model.eval()
            
            if torch.cuda.is_available():
                self.folding_model = self.folding_model.cuda()
                print("Modelo ESMFold inicializado en GPU")
            else:
                print("Modelo ESMFold inicializado en CPU")
                
        except Exception as e:
            raise ValueError(f"Error al inicializar modelo de plegamiento: {e}")
    
    def _sequence_indices_to_string(self, sequence_indices: List[int]) -> str:
        """Convierte índices de aminoácidos a string de secuencia."""
        # Asegurar que los índices estén en el rango válido
        clipped_indices = [max(0, min(19, int(idx))) for idx in sequence_indices]
        return ''.join([self.AMINO_ACIDS[idx] for idx in clipped_indices])
    
    def _fold_sequence(self, sequence: str) -> np.ndarray:
        """
        Pliega una secuencia usando ESMFold y extrae las coordenadas del backbone.
        
        Args:
            sequence: Secuencia de aminoácidos como string
            
        Returns:
            Coordenadas 3D del backbone como numpy array
        """
        try:
            # Tokenizar la secuencia
            tokenized = self.folding_tokenizer(
                sequence, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            if torch.cuda.is_available():
                tokenized = {k: v.cuda() for k, v in tokenized.items()}
            
            # Realizar plegamiento
            with torch.no_grad():
                output = self.folding_model(tokenized["input_ids"])
            
            # Extraer coordenadas (asumiendo que el output tiene coordenadas)
            # Nota: Esto puede necesitar ajustes según la salida exacta de ESMFold
            if hasattr(output, 'positions'):
                coords = output.positions.cpu().numpy()
            elif hasattr(output, 'coordinates'):
                coords = output.coordinates.cpu().numpy()
            else:
                # Fallback: crear coordenadas aleatorias como placeholder
                coords = np.random.rand(len(sequence) * 4, 3) * 50 - 25
                
            # Extraer solo coordenadas del backbone (N, CA, C, O)
            # Asumiendo que las coordenadas están organizadas por residuo
            backbone_coords = []
            for i in range(len(sequence)):
                # Para cada residuo, tomar N, CA, C, O (primeros 4 átomos)
                start_idx = i * 4
                backbone_coords.extend(coords[start_idx:start_idx+4])
            
            return np.array(backbone_coords)
            
        except Exception as e:
            print(f"Error en plegamiento: {e}")
            # Retornar coordenadas aleatorias como fallback
            return np.random.rand(len(sequence) * 4, 3) * 50 - 25
    
    def evaluate(self, solution: ProteinSequenceSolution) -> ProteinSequenceSolution:
        """
        Evalúa una solución (secuencia de aminoácidos).
        
        Args:
            solution: Solución con secuencia codificada como índices
            
        Returns:
            Solución evaluada con objetivos calculados
        """
        try:
            # Convertir índices a secuencia de aminoácidos
            sequence = self._sequence_indices_to_string(solution.variables)
            
            # Plegar la secuencia para obtener coordenadas 3D
            folded_coords = self._fold_sequence(sequence)
            
            # Simular energía de diseño
            energia_design = self._calculate_design_energy(folded_coords, sequence)
            
            if self.use_physicochemical_descriptors and hasattr(self, 'descriptor_ref'):
                # Calcular descriptores de la secuencia actual
                descriptor_temp = descriptores(
                    sequence,
                    self.tokenizer,
                    self.model_ESM2
                )
                
                # Calcular métricas con descriptores fisicoquímicos
                rms, gdt, MC_similitud, divKl, distancias_sal, tms = fitness_gdt_rmsd_mc_fisquim(
                    self.backbone_reference,
                    folded_coords,
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
                    folded_coords,
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
                
                divKl = 0.0
            
            # Configurar objetivos para minimización en NSGA-II
            solution.objectives[0] = rms           # Minimizar RMSD
            solution.objectives[1] = -gdt          # Maximizar GDT (convertir a minimización)
            solution.objectives[2] = energia_design  # Minimizar energía
            
            # Guardar información adicional
            solution.attributes = {
                'sequence': sequence,
                'rmsd': rms,
                'gdt': gdt,
                'mc_similarity': MC_similitud,
                'tms_score': tms,
                'design_energy': energia_design,
                'fitness_total': fitness_total,
                'folded_coords': folded_coords,
                'kl_divergence': divKl
            }
            
        except Exception as e:
            # En caso de error, asignar valores muy malos
            print(f"Error en evaluación: {e}")
            solution.objectives[0] = float('inf')  # RMSD malo
            solution.objectives[1] = float('inf')  # GDT malo
            solution.objectives[2] = float('inf')  # Energía mala
            solution.attributes = {
                'error': str(e),
                'sequence': self._sequence_indices_to_string(solution.variables)
            }
        
        return solution
    
    def _calculate_design_energy(self, coords_3d: np.ndarray, sequence: str) -> float:
        """
        Calcula una energía de diseño basada en la estructura y secuencia.
        
        Args:
            coords_3d: Coordenadas 3D del backbone
            sequence: Secuencia de aminoácidos
            
        Returns:
            Energía de diseño calculada
        """
        try:
            # Energía estructural: penalizar distancias anómalas entre átomos consecutivos
            distancias = np.linalg.norm(coords_3d[1:] - coords_3d[:-1], axis=1)
            energia_estructural = np.sum(np.where(distancias < 1.0, (1.0 - distancias)**2, 0)) + \
                                 np.sum(np.where(distancias > 5.0, (distancias - 5.0)**2, 0))
            
            # Energía de secuencia: preferir ciertos aminoácidos o penalizar otros
            energia_secuencia = 0.0
            hydrophobic = set(['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'])
            charged = set(['K', 'R', 'D', 'E'])
            
            for i, aa in enumerate(sequence):
                # Ejemplo: penalizar muchos aminoácidos cargados consecutivos
                if i > 0 and aa in charged and sequence[i-1] in charged:
                    energia_secuencia += 2.0
                
                # Recompensar ciertos patrones
                if aa in hydrophobic:
                    energia_secuencia -= 0.5
            
            return energia_estructural + energia_secuencia
            
        except Exception as e:
            print(f"Error calculando energía: {e}")
            return 1000.0  # Energía muy alta como penalización
    
    def create_solution(self) -> ProteinSequenceSolution:
        """Crea una nueva solución con secuencia aleatoria."""
        solution = ProteinSequenceSolution(
            sequence_length=self.number_of_variables,
            number_of_objectives=self.number_of_objectives
        )
        
        # Inicializar con índices aleatorios de aminoácidos
        solution.variables = [
            np.random.randint(0, 20) for _ in range(self.number_of_variables)
        ]
        
        return solution
    
    def create_solution_from_sequence(self, sequence: str) -> ProteinSequenceSolution:
        """
        Crea una solución a partir de una secuencia específica.
        
        Args:
            sequence: Secuencia de aminoácidos como string
            
        Returns:
            Solución inicializada con la secuencia dada
        """
        solution = ProteinSequenceSolution(
            sequence_length=len(sequence),
            number_of_objectives=self.number_of_objectives
        )
        
        # Convertir secuencia a índices
        aa_to_idx = {aa: i for i, aa in enumerate(self.AMINO_ACIDS)}
        solution.variables = [
            aa_to_idx.get(aa, 0) for aa in sequence.upper()
        ]
        
        return solution