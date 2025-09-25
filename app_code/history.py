from flwr.server.history import History
from functools import reduce

class MyHistory(History):
    """Extensión de History que añade optimized_M."""

    def __init__(self) -> None:
        super().__init__()
        self.fedMOpt: list[tuple[int, tuple[int, float, float]]] = []

    def add_fedMOpt(self, server_round: int, optimized_M: int, utility_list: float, time_list: float) -> None:
        self.fedMOpt.append((server_round, (optimized_M, utility_list, time_list)))

    def __repr__(self) -> str:
        # obtenemos la representación base
        rep = super().__repr__()

        # añadimos nuestra parte
        if self.fedMOpt:
            rep += f"\nHistory (M, utilities, times):\n" + reduce(
                lambda a, b: a + b,
                [
                    f"\tround {server_round}: {optimized_M, utility_list, time_list}\n"
                    for server_round, (optimized_M, utility_list, time_list) in self.fedMOpt
                ],
            )
        return rep
