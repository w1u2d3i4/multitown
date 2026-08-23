FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash build-essential ca-certificates coreutils curl findutils git golang-go \
    grep jq nodejs npm sed sqlite3 \
 && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.lock.txt /tmp/requirements.lock.txt
RUN python -m pip install --no-cache-dir --requirement /tmp/requirements.lock.txt \
 && rm /tmp/requirements.lock.txt

WORKDIR /workspace
