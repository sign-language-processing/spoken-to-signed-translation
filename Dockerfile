FROM python:3.15.2-slim-trixie@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock /opt/requirements.lock
COPY pyproject.toml README.md ./
COPY spoken_to_signed ./spoken_to_signed
RUN pip install --no-cache-dir -r /opt/requirements.lock && pip install --no-deps '.[server,postgres,gcs]' \
    && useradd --create-home --uid 1000 app

RUN rm -rf /usr/local/lib/python*/ensurepip /usr/local/lib/python*/site-packages/pip /usr/local/lib/python*/site-packages/pip-*.dist-info /opt/venv/lib/python*/site-packages/pip /opt/venv/lib/python*/site-packages/pip-*.dist-info

USER app
ARG MODEL_VERSION
ENV MODEL_VERSION=${MODEL_VERSION} \
    PYTHONUNBUFFERED=1 \
    PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec python -m hypercorn spoken_to_signed.server:app --bind 0.0.0.0:$PORT"]
