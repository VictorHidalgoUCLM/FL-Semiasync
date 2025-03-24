# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower driver app."""


from logging import INFO
from typing import Optional
import io

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import Server, init_defaults
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import Strategy

from flwr.server import Driver
from flwr.server.compat.app_utils import start_update_client_manager_thread

import toml
import os


def start_driver(  # pylint: disable=too-many-arguments, too-many-locals
    *,
    driver: Driver,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
) -> History:
    """Start a Flower Driver API server.

    Parameters
    ----------
    driver : Driver
        The Driver object to use.
    server : Optional[flwr.server.Server] (default: None)
        A server implementation, either `flwr.server.Server` or a subclass
        thereof. If no instance is provided, then `start_driver` will create
        one.
    config : Optional[ServerConfig] (default: None)
        Currently supported values are `num_rounds` (int, default: 1) and
        `round_timeout` in seconds (float, default: None).
    strategy : Optional[flwr.server.Strategy] (default: None).
        An implementation of the abstract base class
        `flwr.server.strategy.Strategy`. If no strategy is provided, then
        `start_server` will use `flwr.server.strategy.FedAvg`.
    client_manager : Optional[flwr.server.ClientManager] (default: None)
        An implementation of the class `flwr.server.ClientManager`. If no
        implementation is provided, then `start_driver` will use
        `flwr.server.SimpleClientManager`.

    Returns
    -------
    hist : flwr.server.history.History
        Object containing training and evaluation metrics.
    """
    # Initialize the Driver API server and config
    initialized_server, initialized_config = init_defaults(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )
    log(
        INFO,
        "Starting Flower ServerApp, config: %s",
        initialized_config,
    )
    log(INFO, "")

    # Start the thread updating nodes
    thread, f_stop = start_update_client_manager_thread(
        driver, initialized_server.client_manager()
    )

    # Start training
    hist = run_fl(
        server=initialized_server,
        config=initialized_config,
    )

    # Terminate the thread
    f_stop.set()
    thread.join()

    return hist


def run_fl(
    server: Server,
    config: ServerConfig,
) -> History:
    """Train a model on the given server and return the History object."""
    hist, elapsed_time = server.fit(
        num_rounds=config.num_rounds, timeout=config.round_timeout
    )

    log(INFO, "")
    log(INFO, "[SUMMARY]")
    log(INFO, "Run finished %s round(s) in %.2fs", config.num_rounds, elapsed_time)
    for line in io.StringIO(str(hist)):
        log(INFO, "\t%s", line.strip("\n"))

    projectconf = toml.load(os.environ.get('CONFIG_PATH'))

    log(INFO, "")
    log(INFO, f"Training {projectconf['tempConfig']['federation']} {projectconf['tempConfig']['strategy']} {projectconf['tempConfig']['execution_name']} {projectconf['tempConfig']['num_exec']} ended")

    # Graceful shutdown
    server.disconnect_all_clients(timeout=config.round_timeout)

    return hist