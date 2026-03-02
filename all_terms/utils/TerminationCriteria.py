import time
import numpy as np
from jmetal.util.termination_criterion import TerminationCriterion
from ..pdb_seq_tools import det_sec
from ..diversity_tools import population_diversity_blosum62


class StoppingByTime(TerminationCriterion):

    def __init__(self, max_seconds: float):
        super().__init__()
        self.max_seconds = max_seconds
        self.start_time = None

    def update(self, algorithm):
        if self.start_time is None:
            self.start_time = time.time()

    @property
    def is_met(self):
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) >= self.max_seconds


class StoppingByDiversity(TerminationCriterion):

    def __init__(self, min_diversity: float, patience: int, amino_seq_ref):
        super().__init__()
        self.min_diversity = min_diversity
        self.patience = patience
        self.counter = 0
        self.amino_seq_ref = amino_seq_ref

    def update(self, algorithm):
        population = algorithm.solutions

        sequences = []
        for sol in population:
            posiciones = [int(round(v)) for v in sol.variables]
            seq = det_sec(posiciones, self.amino_seq_ref)
            sequences.append(seq)

        diversity = population_diversity_blosum62(sequences)

        print(f"Diversidad actual: {diversity:.4f}")

        if diversity < self.min_diversity:
            self.counter += 1
        else:
            self.counter = 0

    @property
    def is_met(self):
        return self.counter >= self.patience


class CombinedTermination:

    def __init__(self, terminations):
        self.terminations = terminations

    def update(self, **kwargs):
        """
        jMetalPy pasa datos como:
        PROBLEM, ALGORITHM, EVALUATIONS, TIME, etc.
        """
        algorithm = kwargs.get("ALGORITHM")

        if algorithm is None:
            raise ValueError("ALGORITHM no fue recibido en kwargs")

        for term in self.terminations:
            term.update(algorithm)

    def is_met(self):
        return any(term.is_met() for term in self.terminations)