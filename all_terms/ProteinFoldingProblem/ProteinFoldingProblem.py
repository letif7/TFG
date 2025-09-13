import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from typing import List, Dict
import logging
from transformers import EsmTokenizer, EsmModel
from ..Fitness.fitness import (
    descriptores, mapa_contacto,
    fitness_gdt_rmsd_mc_fisquim, fitness_gdt_rmsd_mc,
    agrega_rmsd_gdt_E_MC_divKl, agrega_rmsd_gdt_E_MC
)
from ..pdb_seq_tools.pdb_seq_tools import (
    extract_amino_acid_sequence, extract_backbone_atoms_str
)
import pyrosetta
from pyrosetta import pose_from_sequence, get_fa_scorefxn
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.simple_moves import ShakeStructureMover
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.kinematics import MoveMap

logging.basicConfig(level=logging.INFO)

class ProteinFoldingProblem(FloatProblem):
    """
    Optimización multiobjetivo de secuencias de proteínas usando NSGA-II.
    Cada cromosoma representa una secuencia de aminoácidos. Integración con PyRosetta
    y descriptores ESM2 opcionales.
    """

    def __init__(self, 
                 pdb_reference: str,
                 sequence_length: int,
                 use_physicochemical_descriptors: bool = True,
                 corte: List[float] = [1.0, 2.0, 4.0, 8.0],
                 energia_params: tuple = (-50.0, 10.0),
                 init_pyrosetta: bool = True,
                 folding_method: str = "abrelax"):
        
        self.pdb_reference = pdb_reference
        self.sequence_length = sequence_length
        self.use_physicochemical_descriptors = use_physicochemical_descriptors
        self.corte = corte
        self.energia_a, self.energia_b = energia_params
        self.folding_method = folding_method

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

        # Inicializar PyRosetta
        if init_pyrosetta:
            self._initialize_pyrosetta()

        # Inicializar ESM2 si corresponde
        if self.use_physicochemical_descriptors:
            self._initialize_esm2_model()
            self.descriptor_cache = {}  # Cache para secuencias ya evaluadas

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
        return "Protein Sequence Folding Problem"

    def _extract_reference_data(self):
        try:
            self.backbone_reference = extract_backbone_atoms_str(self.pdb_reference)
            self.reference_sequence = extract_amino_acid_sequence(self.pdb_reference)
            self.MC_reference = mapa_contacto(self.backbone_reference)
            logging.info(f"Secuencia de referencia: {self.reference_sequence}")
            if self.sequence_length != len(self.reference_sequence):
                logging.warning(f"Longitud especificada ({self.sequence_length}) difiere de la secuencia de referencia ({len(self.reference_sequence)})")
        except Exception as e:
            raise ValueError(f"Error al extraer datos de referencia del PDB: {e}")

    def _initialize_pyrosetta(self):
        try:
            pyrosetta.init("-mute all")
            self.scorefxn = get_fa_scorefxn()
            self._setup_folding_movers()
            try:
                self.reference_pose = pyrosetta.pose_from_file(self.pdb_reference)
            except:
                self.reference_pose = pose_from_sequence(self.reference_sequence)
                logging.info("Pose generada desde secuencia de referencia")
        except Exception as e:
            raise ValueError(f"Error al inicializar PyRosetta: {e}")

    def _setup_folding_movers(self):
        try:
            if self.folding_method == "fastrelax" or self.folding_method == "abrelax":
                self.folding_mover = FastRelax()
                self.folding_mover.set_scorefxn(self.scorefxn)
            elif self.folding_method == "shake":
                self.folding_mover = ShakeStructureMover()
            else:
                self.folding_mover = FastRelax()
                self.folding_mover.set_scorefxn(self.scorefxn)
        except Exception as e:
            logging.warning(f"Error al configurar movers: {e}")
            self.folding_mover = None

    def _initialize_esm2_model(self):
        try:
            self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.model_ESM2 = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.model_ESM2.eval()
            self.descriptor_ref = descriptores(self.reference_sequence, self.tokenizer, self.model_ESM2)
            logging.info("Modelo ESM2 inicializado correctamente")
        except Exception as e:
            logging.warning(f"Error al inicializar ESM2, continuando sin descriptores: {e}")
            self.use_physicochemical_descriptors = False

    def sequence_from_solution(self, solution: FloatSolution) -> str:
        return ''.join(self.num_to_aa[int(np.clip(round(var),0,19))] for var in solution.variables)

    def fold_sequence(self, sequence: str) -> pyrosetta.Pose:
        try:
            pose = pose_from_sequence(sequence)
            if self.folding_mover is not None:
                self.folding_mover.apply(pose)
            else:
                mm = MoveMap()
                mm.set_bb(True)
                min_mover = MinMover()
                min_mover.movemap(mm)
                min_mover.score_function(self.scorefxn)
                min_mover.min_type('dfpmin')
                min_mover.apply(pose)
            return pose
        except Exception as e:
            logging.warning(f"Error en plegamiento: {e}")
            return pose_from_sequence(sequence)

    def extract_backbone_coordinates(self, pose: pyrosetta.Pose) -> np.ndarray:
        coords = np.zeros((pose.total_residue()*4, 3))
        atom_names = ['N','CA','C','O']
        idx = 0
        for res_idx in range(1, pose.total_residue()+1):
            residue = pose.residue(res_idx)
            for atom_name in atom_names:
                if residue.has(atom_name):
                    xyz = residue.xyz(atom_name)
                    coords[idx] = [xyz.x, xyz.y, xyz.z]
                idx += 1
        return coords

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        try:
            sequence = self.sequence_from_solution(solution)
            folded_pose = self.fold_sequence(sequence)
            coords_3d = self.extract_backbone_coordinates(folded_pose)
            energia_design = self.scorefxn(folded_pose)

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
            solution.objectives = [float('inf')] * self.number_of_objectives
            solution.attributes = {'sequence': self.sequence_from_solution(solution), 'error': str(e)}

        return solution

    def create_solution(self) -> FloatSolution:
        s = FloatSolution(self.lower_bound, self.upper_bound, self.number_of_objectives)
        s.variables = [np.random.uniform(0.0, 19.0) for _ in range(self.number_of_variables)]
        return s

    def create_solution_from_sequence(self, sequence: str) -> FloatSolution:
        s = FloatSolution(self.lower_bound, self.upper_bound, self.number_of_objectives)
        s.variables = [float(self.aa_to_num.get(aa,0)) for aa in sequence[:self.sequence_length]]
        while len(s.variables) < self.sequence_length:
            s.variables.append(np.random.uniform(0.0,19.0))
        return s

    def get_sequence_similarity(self, seq1: str, seq2: str) -> float:
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for a,b in zip(seq1[:min_len], seq2[:min_len]) if a==b)
        return matches / min_len if min_len>0 else 0.0

    def save_solution_pdb(self, solution: FloatSolution, filename: str):
        try:
            sequence = self.sequence_from_solution(solution)
            folded_pose = self.fold_sequence(sequence)
            folded_pose.dump_pdb(filename)
            logging.info(f"Estructura guardada en {filename}")
        except Exception as e:
            logging.error(f"Error al guardar PDB: {e}")

    def get_folding_statistics(self) -> Dict:
        return {
            'sequence_length': self.sequence_length,
            'folding_method': self.folding_method,
            'reference_sequence': self.reference_sequence,
            'use_physicochemical_descriptors': self.use_physicochemical_descriptors,
            'number_of_objectives': self.number_of_objectives,
            'amino_acids': list(self.aa_to_num.keys())
        }
