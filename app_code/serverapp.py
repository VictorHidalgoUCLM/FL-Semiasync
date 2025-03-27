from flwr.server.server_app import ServerApp
from flwr.server import Driver
from flwr.common import Context
from .start_driver import start_driver


class ServerAppCustom(ServerApp):
    """Intermediate class that inherits from ServerApp so it can be used with custom start_driver."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, driver: Driver, context: Context) -> None:
        """Execute `ServerApp`."""
        # Compatibility mode
        if not self._main:
            if self._server_fn:
                # Execute server_fn()
                components = self._server_fn(context)
                self._server = components.server
                self._config = components.config
                self._strategy = components.strategy
                self._client_manager = components.client_manager
            start_driver(
                server=self._server,
                config=self._config,
                strategy=self._strategy,
                client_manager=self._client_manager,
                driver=driver,
            )
            return
