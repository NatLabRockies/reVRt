#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
pixi run -e dev cargo run --release -p revrt-cli --bin rust_route_profile --features="profiling" -- "$@"
