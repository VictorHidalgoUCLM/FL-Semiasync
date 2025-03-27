from typing import Any, Dict, Optional

from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .FedAvg import FedAvgCustom


class FedProxCustom(FedAvgCustom):
    def __init__(
        self,
        parameters_fit: Dict[str, float],
        *args: Any,
        **kwargs: Optional[Dict[str, Any]],
    ):
        """Initialize the FedProxCustom strategy.

        Args:
            parameters_fit: Parameters strategy will use as configuration.
            *args: Additional positional arguments for the superclass.
            **kwargs: Additional keyword arguments for the superclass.
        """

        super().__init__(*args, **kwargs)
        self.parameters = parameters_fit

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedProx(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Sends the proximal factor mu to the clients
        """

        fit_config = super().configure_fit(
            server_round, parameters, client_manager, self.parameters
        )

        # Return client/config pairs with the proximal factor mu added
        return fit_config
