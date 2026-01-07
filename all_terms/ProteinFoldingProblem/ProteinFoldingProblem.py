import os
import numpy as np
import torch
import logging
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from typing import List, Dict
import esm

from transformers import EsmTokenizer, EsmModel, EsmForProteinFolding
from ..pdb_seq_tools.pdb_seq_tools import extract_amino_acid_sequence, extract_backbone_atoms_str
from ..Fitness.fitness import descriptores, mapa_contacto, \
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
        self._number_of_objectives = 7
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
            self.MC_reference = mapa_contacto(self.backbone_reference)

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

    def _calculate_design_energy(self, coords_3d) -> float:
        """Calcula energía de diseño simplificada basada en distancias consecutivas."""

        # asegurar que coords_3d sea un numpy array
        coords_3d = np.asarray(coords_3d, dtype=float)

        # validaciones básicas de forma
        if coords_3d.ndim != 2 or coords_3d.shape[0] < 2 or coords_3d.shape[1] != 3:
            logging.warning(
                f"coords_3d inválido en _calculate_design_energy: shape={coords_3d.shape}"
            )
            # Devolvemos una energía muy alta para penalizar esta solución
            return float("inf")

        # logica original de energía
        distancias = np.linalg.norm(coords_3d[1:] - coords_3d[:-1], axis=1)

        energia_corta = np.sum(
            np.where(distancias < 1.0, (1.0 - distancias) ** 2, 0.0)
        )
        energia_larga = np.sum(
            np.where(distancias > 5.0, (distancias - 5.0) ** 2, 0.0)
        )

        return float(energia_corta + energia_larga)



    # ================================================
    # Evaluar solución NSGA-II
    # =============🔹===================================
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        try:
            sequence = self.sequence_from_solution(solution)
            pdb_str, coords_3d = self.fold_sequence(sequence)

            # Energía con cache
            if sequence in self.energy_cache:
                energia_design = self.energy_cache[sequence]
            else:
                energia_design = self._calculate_design_energy(coords_3d)
                self.energy_cache[sequence] = energia_design

            # Descriptores físico-químicos
            if self.use_physicochemical_descriptors:
                if sequence in self.descriptor_cache:
                    descriptor_temp = self.descriptor_cache[sequence]
                else:
                    descriptor_temp = descriptores(sequence, self.tokenizer, self.model_ESM2)
                    self.descriptor_cache[sequence] = descriptor_temp

                rms, gdt, MC_similitud, divKl, distancias_sal, tms = (
                    fitness_gdt_rmsd_mc_fisquim(
                        self.backbone_reference, coords_3d,
                        self.MC_reference, self.corte,
                        self.descriptor_ref, descriptor_temp
                    )
                )
                fitness_total = agrega_rmsd_gdt_E_MC_divKl(
                    MC_similitud, energia_design, rms, gdt,
                    divKl, self.energia_a, self.energia_b, tms
                )
            else:
                rms, gdt, MC_similitud, distancias_sal, tms = (
                    fitness_gdt_rmsd_mc(
                        self.backbone_reference, coords_3d,
                        self.MC_reference, self.corte
                    )
                )
                fitness_total = agrega_rmsd_gdt_E_MC(
                    MC_similitud, energia_design, rms, gdt,
                    self.energia_a, self.energia_b, tms
                )

                 # Objetivos NSGA-II
                # 1. RMSD (min)
                solution.objectives[0] = rms

                # 2. GDT (max → min)
                solution.objectives[1] = -gdt

                # 3. Energía de diseño (min)
                solution.objectives[2] = energia_design

                # 4. Mapa de contacto (max → min)
                solution.objectives[3] = -MC_similitud

                # 5. Divergencia KL fisicoquímica (min)
                if self.use_physicochemical_descriptors:
                    solution.objectives[4] = divKl
                else:
                    solution.objectives[4] = 0.0  # o np.nan / penalización

                # 6. TM-score (max → min)
                solution.objectives[5] = -tms

                # 7. Distancias salinas / electrostáticas (min)
                solution.objectives[6] = distancias_sal

            solution.attributes = {
                "sequence": sequence,
                "rmsd": rms,
                "gdt": gdt,
                "mc_similarity": MC_similitud,
                "tms_score": tms,
                "design_energy": energia_design,
                "fitness_total": fitness_total,
                "folded_coordinates": coords_3d
            }
            if self.use_physicochemical_descriptors:
                solution.attributes["kl_divergence"] = divKl

        except Exception as e:
            logging.error(f"Error evaluando: {e}")
            #solution.objectives = [float("inf")] * self._number_of_objectives
            #solution.attributes = {"sequence": sequence, "error": str(e)}
            raise

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