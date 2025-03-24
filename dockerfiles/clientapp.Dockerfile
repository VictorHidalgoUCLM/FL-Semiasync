# Dockerfile para imagen personalizada de flwr_clientapp
FROM flwr/clientapp:1.16.0

WORKDIR /app
COPY pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
    && python -m pip install -U .

COPY projectconf.toml .
ENV CONFIG_PATH=/app/projectconf.toml

# Define el punto de entrada del contenedor
ENTRYPOINT ["flwr-clientapp"]
