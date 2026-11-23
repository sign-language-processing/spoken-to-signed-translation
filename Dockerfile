FROM python:3.15.2-slim-trixie AS builder
WORKDIR /app
COPY pyproject.toml README.md ./
COPY spoken_to_signed/ ./spoken_to_signed/
RUN python -m venv --without-pip /opt/venv \
    && python -m pip --python /opt/venv/bin/python install --no-cache-dir ".[server,postgres,gcs]"

FROM python:3.15.2-slim-trixie AS runtime
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* /usr/local/lib/python3.15/ensurepip /usr/local/lib/python3.15/site-packages /usr/local/bin/pip*
RUN useradd --create-home --uid 1000 app
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PORT=8080
COPY --from=builder /opt/venv /opt/venv
COPY spoken_to_signed/ ./spoken_to_signed/
ARG MODEL_VERSION
ENV MODEL_VERSION=${MODEL_VERSION}
USER app
EXPOSE 8080
CMD ["sh", "-c", "exec python -m hypercorn spoken_to_signed.server:app --bind 0.0.0.0:$PORT"]
