FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY spoken_to_signed ./spoken_to_signed
RUN pip install --no-cache-dir '.[server,postgres,gcs]' \
    && useradd --create-home --uid 1000 app

USER app
ARG MODEL_VERSION
ENV MODEL_VERSION=${MODEL_VERSION} \
    PYTHONUNBUFFERED=1 \
    PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec python -m uvicorn spoken_to_signed.server:app --host 0.0.0.0 --port $PORT"]
