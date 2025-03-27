from typing import Any, Dict, Optional

from flwr.common import (
    NDArrays,
    parameters_to_ndarrays,
)

from .FedAvg import FedAvgCustom


class FedOptCustom(FedAvgCustom):
    def __init__(
        self,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        tau: float = 1e-9,
        beta_1: float = 0.0,
        beta_2: float = 0.0,
        *args: Any,
        **kwargs: Optional[Dict[str, Any]],
    ):
        """Initialize the FedAdamCustom strategy.

        Args:
            eta: The global learning rate. Default is 1e-1.
            eta_l: The local learning rate. Default is 1e-1.
            tau: The threshold value for adaptive updates. Default is 1e-9.
            beta_1: First moment decay rate. Default is 0.0.
            beta_2: Second moment decay rate. Default is 0.0.
            *args: Additional positional arguments for the superclass.
            **kwargs: Additional keyword arguments for the superclass.
        """

        super().__init__(*args, **kwargs)
        self.current_weights = parameters_to_ndarrays(self.initial_parameters)
        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_t: Optional[NDArrays] = None
        self.v_t: Optional[NDArrays] = None

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedOpt(accept_failures={self.accept_failures})"
        return rep
