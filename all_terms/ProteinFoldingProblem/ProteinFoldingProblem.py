import os
import time
import numpy as np
import torch
import logging
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from typing import List, Dict
import esm
from pyrosetta import Pose
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
from pyrosetta.rosetta.core.scoring import get_score_function

from transformers import EsmTokenizer, EsmModel, EsmForProteinFolding
from ..pdb_seq_tools.pdb_seq_tools import extract_amino_acid_sequence, extract_backbone_atoms_str
from ..Fitness.fitness import descriptores, mapa_contacto_distancias, mapa_contacto_binario, \
    fitness_gdt_rmsd_mc_fisquim, fitness_gdt_rmsd_mc, \
    agrega_rmsd_gdt_E_MC_divKl, agrega_rmsd_gdt_E_MC

logging.basicConfig(level=logging.INFO)

class ProteinFoldingProblem(FloatProblem):
    """
    Optimización multiobjetivo de secuencias proteicas
    usando NSGA-II + ESM-Fold + energía simplificada.
    """

    def __init__(
        self,
        pdb_reference: str,
        sequence_length: int,
        use_physicochemical_descriptors: bool = True,
        corte: List[float] = [1.0, 2.0, 4.0, 8.0],
        energia_params: tuple = (-50.0, 10.0)
    ):

        self.pdb_reference = pdb_reference
        self.sequence_length = sequence_length
        self.use_physicochemical_descriptors = use_physicochemical_descriptors
        self.corte = corte
        self.energia_a, self.energia_b = energia_params

        # Mapas AA → número
        self.aa_to_num = {aa: i for i, aa in enumerate(
            ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        )}
        self.num_to_aa = {v: k for k, v in self.aa_to_num.items()}

        # Propiedades JMetalPy
        self._number_of_variables = sequence_length
        self._number_of_objectives = 3
        self._number_of_constraints = 0
        self._lower_bound = [0.0] * sequence_length
        self._upper_bound = [19.0] * sequence_length

 

        # Extrae referencia
        self._extract_reference_data()

        # Inicializa modelos ESM2 + ESM-Fold
        self._initialize_esm_models()

        # Cachés
        self.descriptor_cache = {}
        self.energy_cache = {}


    # ================================================
    # Extracción de datos de PDB de referencia
    # ================================================
    def number_of_objectives(self):
            return self._number_of_objectives
    
    def _extract_reference_data(self):
        try:
            if isinstance(self.pdb_reference, str) and os.path.exists(self.pdb_reference):
                pdb_text = open(self.pdb_reference).read()
            else:
                pdb_text = self.pdb_reference

            self.backbone_reference = extract_backbone_atoms_str(pdb_text)
            self.reference_sequence = extract_amino_acid_sequence(pdb_text)
            self.MC_reference = mapa_contacto_distancias(self.backbone_reference)
            self.mapa_binario_referencia_MC = mapa_contacto_binario(self.MC_reference)


            logging.info(f"Secuencia referencia: {self.reference_sequence}")

            if self.sequence_length != len(self.reference_sequence):
                logging.warning(
                    f"Longitud especificada {self.sequence_length} diferente a la referencia {len(self.reference_sequence)}"
                )

        except Exception as e:
            raise ValueError(f"Error extrayendo referencia del PDB: {e}")


    # ================================================
    # Inicialización ESM
    # ================================================
    def _initialize_esm_models(self):
        try:
            logging.info("Inicializando ESM2 + ESM-Fold...")

            self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.model_ESM2 = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").eval()
            self.esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
            if torch.cuda.is_available():
                self.esmfold_model = self.esmfold_model.cuda()

            self.descriptor_ref = descriptores(self.reference_sequence, self.tokenizer, self.model_ESM2)

        except Exception as e:
            raise ValueError(f"Error inicializando ESM: {e}")


    # ================================================
    # Secuencia → string de aminoácidos
    # ================================================
    def sequence_from_solution(self, solution: FloatSolution) -> str:
        return ''.join(
            self.num_to_aa[int(np.clip(round(v),0,19))]
            for v in solution.variables
        )


    # ================================================
    # ESM-FOLD: Secuencia → PDB + coords 3D
    # ================================================
    def fold_sequence(self, sequence: str):
        try:
            with torch.no_grad():
                pdb_str = self.esmfold_model.infer_pdb(sequence)
            coords = extract_backbone_atoms_str(pdb_str)
            return pdb_str, coords
        except Exception as e:
            logging.error(f"Error en ESM-Fold: {e}")
            return "", np.zeros((self.sequence_length * 4, 3))


    # ================================================
    # Energía simplificada
    # ================================================
    import numpy as np
    import logging

    def calcular_energia_pyrosetta(self, pdb_str: str) -> float:
        try:
            pose = Pose()
            pose_from_pdbstring(pose, pdb_str)

            sfxn = get_score_function(True)
            energia = sfxn(pose)
            return float(energia)

        except Exception as e:
            logging.warning(f"Error calculando energía PyRosetta: {e}")
            return float("inf")

    def _to_float(self, x, default=float("inf")):
        import numpy as np
        if x is None:
            return float(default)
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        if isinstance(x, np.ndarray):
            # si es array, intentamos convertir a escalar
            if x.size == 1:
                return float(x.item())
            # si es vector, lo reducimos (ej. suma). Ajustá según tu métrica.
            return float(np.sum(x))
        if isinstance(x, (list, tuple)):
            # si es lista, la reducimos
            return float(np.sum(x))
        return float(default)


    # ================================================
    # Evaluar solución NSGA-II
    # =============🔹===================================
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        t0 = time.time()
        try:
            sequence = self.sequence_from_solution(solution)
            pdb_str, coords_3d = self.fold_sequence(sequence)

            # Energía desde la estructura plegada por ESMFold ✅
            energia_rosetta = self.calcular_energia_pyrosetta(pdb_str)

            if sequence in self.descriptor_cache:
                descriptor_temp = self.descriptor_cache[sequence]
            else:
                descriptor_temp = descriptores(sequence, self.tokenizer, self.model_ESM2) #obtiene una representacion estadistica de la secuencia y de eso deriva los rasgos fisico-quimicos/probabilisticos
                self.descriptor_cache[sequence] = descriptor_temp

            rms, gdt, MC_similitud, divKl, tms = (
                fitness_gdt_rmsd_mc_fisquim(
                    self.backbone_reference, coords_3d, sequence, self.reference_sequence ,
                    self.mapa_binario_referencia_MC, self.corte,
                    self.descriptor_ref, descriptor_temp
                )
            )
            f1, f2, f3 = agrega_rmsd_gdt_E_MC_divKl(
                MC_similitud,  rms, gdt,
                divKl, energia_rosetta, 30, tms
            )
            

            # Objetivos NSGA-II
            solution.objectives[0] = self._to_float(f1)

            solution.objectives[1] = self._to_float(f2)

            solution.objectives[2] = self._to_float(f3)

            solution.attributes = {
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "sequence": sequence,
                "rmsd": rms,
                "gdt": gdt,
                "mc_similarity": MC_similitud,
                "tms_score": tms,
                "design_energy": energia_design_b,
                "folded_coordinates": coords_3d
            }

        except Exception as e:
            logging.error(f"Error evaluando: {e}")
            #solution.objectives = [float("inf")] * self._number_of_objectives
            #solution.attributes = {"sequence": sequence, "error": str(e)}
            raise
        print("eval_s:", round(time.time() - t0, 3))
        return solution


    # ================================================
    # Crear solución aleatoria
    # ================================================
    def create_solution(self) -> FloatSolution:
        s = FloatSolution(self._lower_bound, self._upper_bound, self._number_of_objectives)
        s.variables = [np.random.uniform(0,19) for _ in range(self.sequence_length)]
        return s


    # ================================================
    # Crear solución a partir de secuencia
    # ================================================
    def create_solution_from_sequence(self, sequence: str) -> FloatSolution:
        s = FloatSolution(self._lower_bound, self._upper_bound, self._number_of_objectives)
        s.variables = [float(self.aa_to_num[x]) for x in sequence]
        return s


    # ================================================
    # Información final
    # ================================================
    def get_folding_statistics(self) -> Dict:
        return {
            "sequence_length": self.sequence_length,
            "reference_sequence": self.reference_sequence,
            "objectives": self._number_of_objectives,
            "aa_list": list(self.aa_to_num.keys())
        }

        # ================================================
    # Implementación requerida por jMetalPy (FloatProblem)
    # ================================================
    @property
    def number_of_variables(self) -> int:
        return self._number_of_variables

    

    @property
    def number_of_constraints(self) -> int:
        return self._number_of_constraints

    def name(self) -> str:
        return "ProteinFoldingProblem"