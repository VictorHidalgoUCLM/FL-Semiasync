from typing import Any, Dict, Optional, Union

import numpy as np

from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from .FedOpt import FedOptCustom


class FedAdagradCustom(FedOptCustom):
    def __init__(
        self,
        eta: float = 1e-2,
        eta_l: float = 0.0316,
        tau: float = 1e-3,
        *args: Any,
        **kwargs: Optional[Dict[str, Any]],
    ):
        """Initialize the FedAdagradCustom strategy.

        Args:
            eta: The global learning rate. Default is 1e-2.
            eta_l: The local learning rate. Default is 0.0316.
            tau: The threshold value for adaptive updates. Default is 1e-3.
            *args: Additional positional arguments for the superclass.
            **kwargs: Additional keyword arguments for the superclass.
        """

        super().__init__(eta=eta, eta_l=eta_l, tau=tau, *args, **kwargs)

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAdagrad(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)

        # Adagrad
        delta_t: NDArrays = [
            x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
        ]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
            for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [x + np.multiply(y, y) for x, y in zip(self.v_t, delta_t)]

        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
        ]

        self.current_weights = new_weights

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated
