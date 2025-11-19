import numpy as np
import logging
import os
import torch
import tempfile
from typing import List, Dict

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

# -----------------------------
#      ESM-FOLD IMPORTS
# -----------------------------
from transformers import EsmTokenizer, EsmModel, EsmForProteinFolding

# -----------------------------
#      PYROSETTA IMPORTS
# -----------------------------
from pyrosetta.distributed import init as pyrosetta_init
from pyrosetta import (
    pose_from_file,
    get_fa_scorefxn,
    MoveMap,
    MinMover,
)
from pyrosetta.rosetta.protocols.relax import FastRelax

# -----------------------------
#     CUSTOM FITNESS IMPORTS
# -----------------------------
from ..Fitness.fitness import (
    descriptores, mapa_contacto,
    fitness_gdt_rmsd_mc_fisquim, fitness_gdt_rmsd_mc,
    agrega_rmsd_gdt_E_MC_divKl, agrega_rmsd_gdt_E_MC
)

from ..pdb_seq_tools.pdb_seq_tools import (
    extract_amino_acid_sequence, extract_backbone_atoms_str
)

logging.basicConfig(level=logging.INFO)



# ============================================================
#           PROTEIN FOLDING PROBLEM (NSGA-II)
# ============================================================
class ProteinFoldingProblem(FloatProblem):
    """
    Optimización multiobjetivo de secuencias proteicas
    usando NSGA-II + ESM-Fold + PyRosetta.
    """

    def __init__(
        self,
        pdb_reference: str,
        sequence_length: int,
        use_physicochemical_descriptors: bool = True,
        corte: List[float] = [1.0, 2.0, 4.0, 8.0],
        energia_params: tuple = (-50.0, 10.0),
        init_pyrosetta: bool = True,
        do_minimize: bool = True,
        do_relax: bool = False
    ):

        self.pdb_reference = pdb_reference
        self.sequence_length = sequence_length
        self.use_physicochemical_descriptors = use_physicochemical_descriptors
        self.corte = corte
        self.energia_a, self.energia_b = energia_params
        self.do_minimize = do_minimize
        self.do_relax = do_relax

        # Mapas AA → número
        self.aa_to_num = {aa: i for i, aa in enumerate(
            ['A','R','N','D','C','Q','E','G',
             'H','I','L','K','M','F','P','S',
             'T','W','Y','V']
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

        # Inicializa PyRosetta
        if init_pyrosetta:
            self._initialize_pyrosetta()

        # Cachés
        self.descriptor_cache = {}
        self.energy_cache = {}



    # ============================================================
    #                    PDB REFERENCE
    # ============================================================
    def _extract_reference_data(self):
        try:
            if os.path.exists(self.pdb_reference):
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



    # ============================================================
    #                INITIALIZE ESM MODELS
    # ============================================================
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



    # ============================================================
    #                INITIALIZE PYROSETTA
    # ============================================================
    def _initialize_pyrosetta(self):
        try:
            pyrosetta_init("-mute all")
            self.scorefxn = get_fa_scorefxn()
            logging.info("PyRosetta inicializado.")
        except Exception as e:
            raise ValueError(f"Error inicializando PyRosetta: {e}")



    # ============================================================
    #                SECUENCIA → STRING AA
    # ============================================================
    def sequence_from_solution(self, solution: FloatSolution) -> str:
        return ''.join(
            self.num_to_aa[int(np.clip(round(v),0,19))]
            for v in solution.variables
        )



    # ============================================================
    #                ESM-FOLD: SECUENCIA → PDB
    # ============================================================
    def fold_sequence(self, sequence: str):
        try:
            with torch.no_grad():
                pdb_str = self.esmfold_model.infer_pdb(sequence)

            coords = extract_backbone_atoms_str(pdb_str)
            return pdb_str, coords

        except Exception as e:
            logging.error(f"Error en ESM-Fold: {e}")
            return "", np.zeros((self.sequence_length * 4, 3))



    # ============================================================
    #                ENERGY WITH PYROSETTA
    # ============================================================
    def compute_energy_from_pdb(self, pdb_str: str) -> float:
        """Crea pose, minimiza (opcional) y calcula energía."""

        # Guardar PDB temporal
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(pdb_str.encode())

        try:
            pose = pose_from_file(tmp_path)

            # ---- Minimización ----
            if self.do_minimize:
                mm = MoveMap()
                mm.set_bb(True)

                min_mover = MinMover()
                min_mover.movemap(mm)
                min_mover.score_function(self.scorefxn)
                min_mover.min_type('dfpmin')

                try:
                    min_mover.apply(pose)
                except:
                    logging.warning("Minimización falló.")

            # ---- Relax opcional ----
            if self.do_relax:
                relax = FastRelax()
                relax.set_scorefxn(self.scorefxn)
                try:
                    relax.apply(pose)
                except:
                    logging.warning("FastRelax falló.")

            energy = float(self.scorefxn(pose))

            os.remove(tmp_path)
            return energy

        except Exception as e:
            logging.error(f"Error calculando energía: {e}")
            try: os.remove(tmp_path)
            except: pass
            return float("inf")



    # ============================================================
    #               EVALUATE (OBJETIVOS NSGA-II)
    # ============================================================
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        try:
            sequence = self.sequence_from_solution(solution)

            pdb_str, coords_3d = self.fold_sequence(sequence)

            # ----- Energía con cache -----
            if sequence in self.energy_cache:
                energia_design = self.energy_cache[sequence]
            else:
                energia_design = self.compute_energy_from_pdb(pdb_str)
                self.energy_cache[sequence] = energia_design

            # ----- Descriptores físico-químicos -----
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

            # ---- Objetivos NSGA-II ----
            solution.objectives[0] = rms
            solution.objectives[1] = -gdt
            solution.objectives[2] = energia_design

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
            solution.objectives = [float("inf")] * self._number_of_objectives
            solution.attributes = {"sequence": sequence, "error": str(e)}

        return solution



    # ============================================================
    #                RANDOM SOLUTION
    # ============================================================
    def create_solution(self) -> FloatSolution:
        s = FloatSolution(self._lower_bound, self._upper_bound, self._number_of_objectives)
        s.variables = [np.random.uniform(0,19) for _ in range(self.sequence_length)]
        return s



    # ============================================================
    #     CREATE SOLUTION FROM SEQUENCE STRING
    # ============================================================
    def create_solution_from_sequence(self, sequence: str) -> FloatSolution:
        s = FloatSolution(self._lower_bound, self._upper_bound, self._number_of_objectives)
        s.variables = [float(self.aa_to_num[x]) for x in sequence]
        return s



    # ============================================================
    #     FINAL INFO
    # ============================================================
    def get_folding_statistics(self) -> Dict:
        return {
            "sequence_length": self.sequence_length,
            "reference_sequence": self.reference_sequence,
            "objectives": self._number_of_objectives,
            "aa_list": list(self.aa_to_num.keys())
        }
