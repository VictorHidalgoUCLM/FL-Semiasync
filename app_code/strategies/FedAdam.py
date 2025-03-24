from typing import Optional, Union

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

class FedAdamCustom(FedOptCustom):
    def __init__(self,
                 eta: float = 1e-1,
                 eta_l: float = 1e-1,
                 beta_1: float = 0.9,
                 beta_2: float = 0.99,
                 tau: float = 1e-9,
                 *args,
                 **kwargs):
        super().__init__(eta=eta,
                         eta_l=eta_l,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         tau=tau,
                         *args,
                         **kwargs)

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAdam(accept_failures={self.accept_failures})"
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

        # Adam
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
        self.v_t = [
            self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y)
            for x, y in zip(self.v_t, delta_t)
        ]

        # Compute the bias-corrected learning rate, `eta_norm` for improving convergence
        # in the early rounds of FL training. This `eta_norm` is `\alpha_t` in Kingma &
        # Ba, 2014 (http://arxiv.org/abs/1412.6980) "Adam: A Method for Stochastic
        # Optimization" in the formula line right before Section 2.1.
        eta_norm = (
            self.eta
            * np.sqrt(1 - np.power(self.beta_2, server_round + 1.0))
            / (1 - np.power(self.beta_1, server_round + 1.0))
        )

        new_weights = [
            x + eta_norm * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
        ]

        self.current_weights = new_weights

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated