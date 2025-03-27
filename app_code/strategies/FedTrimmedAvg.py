from logging import WARNING
from typing import Any, Dict, Optional, Union

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate_trimmed_avg
from .FedAvg import FedAvgCustom


class FedTrimmedAvgCustom(FedAvgCustom):
    def __init__(
        self, beta: float = 0.2, *args: Any, **kwargs: Optional[Dict[str, any]]
    ):
        """Configurable FedMedian strategy implementation.

        Args:
            beta: Fraction of values to trim.
        """

        super().__init__(*args, **kwargs)
        self.beta = beta

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedTrimmedAvg(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using trimmed average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(
            aggregate_trimmed_avg(weights_results, self.beta)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
