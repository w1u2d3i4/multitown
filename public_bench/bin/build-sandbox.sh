#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build \
  --file "$project_root/docker/runner.Dockerfile" \
  --tag general-mas-runner:0.1 \
  "$project_root"

docker image inspect general-mas-runner:0.1 --format '{{.Id}}'
