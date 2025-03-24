# Dockerfile para imagen personalizada de flwr_superexec
FROM flwr/serverapp:1.16.0

WORKDIR /app
COPY pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
   && python -m pip install -U .

COPY app_code/strategies/ ./app_code/strategies/
ENV CONFIG_PATH=/app/projectconf.toml

USER root
RUN usermod -aG ubuntu app
USER app

# Define el punto de entrada del contenedor
ENTRYPOINT ["flwr-serverapp"]