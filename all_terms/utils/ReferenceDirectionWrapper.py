## utils.ReferenceDirectionsWrapper

class ReferenceDirectionsWrapper:
    """
    Wrapper para listas de reference directions que agrega el método compute(),
    que es lo que espera NSGAIII.
    """
    def __init__(self, directions):
        self._directions = directions

    def compute(self):
        return self._directions
