# #!/usr/bin/env bash
# set -euo pipefail

# # Simple sanity-check runner for graph discovery methods, including AutoCD.
# # Assumes you are in the repo root (where experiments/ lives).
# #
# # Usage:
# #   bash experiments/graph_discovery/run_sanity_all_methods.sh

# Ensure AutoCD package can be imported as `AutoCD.*`
export PYTHONPATH="experiments/graph_discovery/methods:${PYTHONPATH:-}"

# python -m experiments.graph_discovery.run_experiment \
#   --config experiments/graph_discovery/graph_discovery_experiments.yaml \
#   --experiment all_methods_synthetic \
#   --save-interval 0


# python -m experiments.graph_discovery.run_experiment --config experiments/graph_discovery/graph_discovery_experiments.yaml --experiment testing_methods
python -m experiments.graph_discovery.run_experiment --config experiments/graph_discovery/graph_discovery_experiments.yaml --experiment full_synthetic_cd