from typing import Any, Dict, Optional, Union

import numpy as np

from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate_qffl
from .FedAvg import FedAvgCustom


class QFedAvgCustom(FedAvgCustom):
    def __init__(
        self,
        parameters_fit: dict = {"evaluate_on_fit": True},
        q_param: float = 0.2,
        qffl_learning_rate: float = 0.1,
        *args: Any,
        **kwargs: Optional[Dict[str, Any]],
    ):
        """
        Initializes the QFedAvgCustom strategy for federated learning.

        Args:
            parameters_fit: A dictionary containing the configuration for the fitting process.
            q_param: A parameter controlling the custom aggregation process.
            qffl_learning_rate: The learning rate for the QFedAvg strategy.
            *args: Additional positional arguments for the superclass.
            **kwargs: Additional keyword arguments for the superclass.
        """

        super().__init__(*args, **kwargs)
        self.learning_rate = qffl_learning_rate
        self.q_param = q_param
        self.parameters = parameters_fit
        self.pre_weights: Optional[NDArrays] = None

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"QffedAvg(learning_rate={self.learning_rate}, "
        rep += f"q_param={self.q_param}, pre_weights={self.pre_weights})"
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        fit_config = super().configure_fit(server_round, parameters, client_manager)

        weights = parameters_to_ndarrays(parameters)
        self.pre_weights = weights

        # Return client/config pairs
        return fit_config

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        _, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        def norm_grad(grad_list: NDArrays) -> float:
            # input: nested gradients
            # output: square of the L-2 norm
            client_grads = grad_list[0]
            for i in range(1, len(grad_list)):
                client_grads = np.append(
                    client_grads, grad_list[i]
                )  # output a flattened array
            squared = np.square(client_grads)
            summed = np.sum(squared)
            return float(summed)

        deltas = []
        hs_ffl = []

        if self.pre_weights is None:
            raise AttributeError("QffedAvg pre_weights are None in aggregate_fit")

        weights_before = self.pre_weights
        losses = []
        examples = []

        for _, fit_res in results:
            losses.append(fit_res.num_examples * fit_res.metrics["loss"])
            examples.append(fit_res.num_examples)

        loss = sum(losses) / sum(examples)

        for _, fit_res in results:
            new_weights = parameters_to_ndarrays(fit_res.parameters)
            # plug in the weight updates into the gradient
            grads = [
                np.multiply((u - v), 1.0 / self.learning_rate)
                for u, v in zip(weights_before, new_weights)
            ]
            deltas.append(
                [np.float_power(loss + 1e-10, self.q_param) * grad for grad in grads]
            )
            # estimation of the local Lipschitz constant
            hs_ffl.append(
                self.q_param
                * np.float_power(loss + 1e-10, (self.q_param - 1))
                * norm_grad(grads)
                + (1.0 / self.learning_rate)
                * np.float_power(loss + 1e-10, self.q_param)
            )

        weights_aggregated: NDArrays = aggregate_qffl(weights_before, deltas, hs_ffl)
        aggregated_parameters = ndarrays_to_parameters(weights_aggregated)

        return aggregated_parameters, aggregated_metrics
