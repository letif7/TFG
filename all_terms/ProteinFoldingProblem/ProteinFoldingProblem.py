import numpy as np
import logging
import os
import torch
import tempfile
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from typing import List, Dict
from transformers import EsmTokenizer, EsmModel, EsmForProteinFolding

import pyrosetta
from pyrosetta import pose_from_file, get_fa_scorefxn
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.kinematics import MoveMap

from ..Fitness.fitness import (
    descriptores, mapa_contacto,
    fitness_gdt_rmsd_mc_fisquim, fitness_gdt_rmsd_mc,
    agrega_rmsd_gdt_E_MC_divKl, agrega_rmsd_gdt_E_MC
)
from ..pdb_seq_tools.pdb_seq_tools import (
    extract_amino_acid_sequence, extract_backbone_atoms_str
)

logging.basicConfig(level=logging.INFO)

class ProteinFoldingProblem(FloatProblem):
    """
    Optimización multiobjetivo de secuencias de proteínas usando NSGA-II.
    Usa ESM-Fold para generar estructuras 3D y PyRosetta para calcular energía.
    """

    def __init__(self, 
                 pdb_reference: str,
                 sequence_length: int,
                 use_physicochemical_descriptors: bool = True,
                 corte: List[float] = [1.0, 2.0, 4.0, 8.0],
                 energia_params: tuple = (-50.0, 10.0),
                 init_pyrosetta: bool = True,
                 do_minimize: bool = True,
                 do_relax: bool = False):
        
        self.pdb_reference = pdb_reference
        self.sequence_length = sequence_length
        self.use_physicochemical_descriptors = use_physicochemical_descriptors
        self.corte = corte
        self.energia_a, self.energia_b = energia_params
        self.do_minimize = do_minimize
        self.do_relax = do_relax

        # Mapas de aminoácidos
        self.aa_to_num = {aa: i for i, aa in enumerate(
            ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
             'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 
             'T', 'W', 'Y', 'V']
        )}
        self.num_to_aa = {v: k for k, v in self.aa_to_num.items()}

        # Propiedades del problema
        self._number_of_variables = sequence_length
        self._number_of_objectives = 3
        self._number_of_constraints = 0
        self._lower_bound = [0.0] * sequence_length
        self._upper_bound = [19.0] * sequence_length

        # Extraer datos de referencia
        self._extract_reference_data()

        # Inicializar modelos
        self._initialize_esm_models()

        # Inicializar PyRosetta
        if init_pyrosetta:
            self._initialize_pyrosetta()

        # Cachés
        self.descriptor_cache = {}
        self.energy_cache = {}

    def _extract_reference_data(self):
        try:
            if os.path.exists(self.pdb_reference):
                with open(self.pdb_reference, "r") as f:
                    pdb_text = f.read()
            else:
                pdb_text = self.pdb_reference

            self.backbone_reference = extract_backbone_atoms_str(pdb_text)
            self.reference_sequence = extract_amino_acid_sequence(pdb_text)
            self.MC_reference = mapa_contacto(self.backbone_reference)
            logging.info(f"Secuencia de referencia: {self.reference_sequence}")

            if self.sequence_length != len(self.reference_sequence):
                logging.warning(
                    f"Longitud especificada ({self.sequence_length}) difiere de la secuencia de referencia ({len(self.reference_sequence)})"
                )
        except Exception as e:
            raise ValueError(f"Error al extraer datos de referencia del PDB: {e}")

    def _initialize_esm_models(self):
        try:
            logging.info("Inicializando modelos ESM...")
            self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.model_ESM2 = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").eval()
            self.esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
            if torch.cuda.is_available():
                self.esmfold_model = self.esmfold_model.cuda()
            self.descriptor_ref = descriptores(self.reference_sequence, self.tokenizer, self.model_ESM2)
            logging.info("Modelos ESM2 y ESM-Fold inicializados correctamente.")
        except Exception as e:
            raise ValueError(f"Error al inicializar modelos ESM: {e}")

    def _initialize_pyrosetta(self):
        try:
            pyrosetta.init("-mute all")
            self.scorefxn = get_fa_scorefxn()
            logging.info("PyRosetta inicializado correctamente.")
        except Exception as e:
            raise ValueError(f"Error al inicializar PyRosetta: {e}")

    def sequence_from_solution(self, solution: FloatSolution) -> str:
        return ''.join(self.num_to_aa[int(np.clip(round(var), 0, 19))] for var in solution.variables)

    def fold_sequence(self, sequence: str) -> (str, np.ndarray):
        """Genera estructura 3D con ESM-Fold y devuelve PDB string + coordenadas."""
        try:
            with torch.no_grad():
                pdb_str = self.esmfold_model.infer_pdb(sequence)
            coords = extract_backbone_atoms_str(pdb_str)
            return pdb_str, coords
        except Exception as e:
            logging.error(f"Error en predicción con ESM-Fold: {e}")
            return "", np.zeros((self.sequence_length * 4, 3))

    def compute_energy_from_pdb(self, pdb_str: str) -> float:
        """Calcula energía usando PyRosetta a partir de un PDB string."""
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(pdb_str.encode())
        try:
            pose = pose_from_file(tmp_path)
            if self.do_minimize:
                mm = MoveMap()
                mm.set_bb(True)
                min_mover = MinMover()
                min_mover.movemap(mm)
                min_mover.score_function(self.scorefxn)
                min_mover.min_type('dfpmin')
                try:
                    min_mover.apply(pose)
                except Exception as e:
                    logging.warning(f"Minimización falló: {e}")
            if self.do_relax:
                relax = FastRelax()
                relax.set_scorefxn(self.scorefxn)
                try:
                    relax.apply(pose)
                except Exception as e:
                    logging.warning(f"FastRelax falló: {e}")
            energy = float(self.scorefxn(pose))
            os.remove(tmp_path)
            return energy
        except Exception as e:
            logging.error(f"Error al calcular energía: {e}")
            try: os.remove(tmp_path)
            except: pass
            return float('inf')

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        try:
            sequence = self.sequence_from_solution(solution)
            pdb_str, coords_3d = self.fold_sequence(sequence)

            # Energía (usa cache si existe)
            if sequence in self.energy_cache:
                energia_design = self.energy_cache[sequence]
            else:
                energia_design = self.compute_energy_from_pdb(pdb_str)
                self.energy_cache[sequence] = energia_design

            # Fitness estructural
            if self.use_physicochemical_descriptors:
                if sequence in self.descriptor_cache:
                    descriptor_temp = self.descriptor_cache[sequence]
                else:
                    descriptor_temp = descriptores(sequence, self.tokenizer, self.model_ESM2)
                    self.descriptor_cache[sequence] = descriptor_temp

                rms, gdt, MC_similitud, divKl, distancias_sal, tms = fitness_gdt_rmsd_mc_fisquim(
                    self.backbone_reference, coords_3d, self.MC_reference, self.corte,
                    self.descriptor_ref, descriptor_temp
                )
                fitness_total = agrega_rmsd_gdt_E_MC_divKl(
                    MC_similitud, energia_design, rms, gdt, divKl, self.energia_a, self.energia_b, tms
                )
            else:
                rms, gdt, MC_similitud, distancias_sal, tms = fitness_gdt_rmsd_mc(
                    self.backbone_reference, coords_3d, self.MC_reference, self.corte
                )
                fitness_total = agrega_rmsd_gdt_E_MC(
                    MC_similitud, energia_design, rms, gdt, self.energia_a, self.energia_b, tms
                )

            solution.objectives[0] = rms
            solution.objectives[1] = -gdt
            solution.objectives[2] = energia_design

            solution.attributes = {
                'sequence': sequence,
                'rmsd': rms,
                'gdt': gdt,
                'mc_similarity': MC_similitud,
                'tms_score': tms,
                'design_energy': energia_design,
                'fitness_total': fitness_total,
                'folded_coordinates': coords_3d
            }
            if self.use_physicochemical_descriptors:
                solution.attributes['kl_divergence'] = divKl

        except Exception as e:
            logging.error(f"Error en evaluación: {e}")
            solution.objectives = [float('inf')] * self._number_of_objectives
            solution.attributes = {'sequence': self.sequence_from_solution(solution), 'error': str(e)}
        return solution

    def create_solution(self) -> FloatSolution:
        s = FloatSolution(self._lower_bound, self._upper_bound, self._number_of_objectives)
        s.variables = [np.random.uniform(0.0, 19.0) for _ in range(self._number_of_variables)]
        return s

    def create_solution_from_sequence(self, sequence: str) -> FloatSolution:
        s = FloatSolution(self._lower_bound, self._upper_bound, self._number_of_objectives)
        s.variables = [float(self.aa_to_num.get(aa,0)) for aa in sequence[:self.sequence_length]]
        while len(s.variables) < self.sequence_length:
            s.variables.append(np.random.uniform(0.0,19.0))
        return s

    def get_sequence_similarity(self, seq1: str, seq2: str) -> float:
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for a,b in zip(seq1[:min_len], seq2[:min_len]) if a==b)
        return matches / min_len if min_len>0 else 0.0

    def get_folding_statistics(self) -> Dict:
        return {
            'sequence_length': self.sequence_length,
            'reference_sequence': self.reference_sequence,
            'use_physicochemical_descriptors': self.use_physicochemical_descriptors,
            'number_of_objectives': self._number_of_objectives,
            'amino_acids': list(self.aa_to_num.keys())
        }
