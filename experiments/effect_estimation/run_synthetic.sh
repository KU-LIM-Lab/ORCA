# Run experiments for effect estimation
# idhp oracle, ci oracle, ci agentgraph
# how to run: bash experiments/effect_estimation/run.sh

# python -m experiments.effect_estimation.run_experiment --config experiments/effect_estimation/effect_experiments.yaml --experiment ihdp_oracle
python -m experiments.effect_estimation.run_experiment --config experiments/effect_estimation/effect_experiments.yaml --experiment ci_oracle
# python -m experiments.effect_estimation.run_experiment --config experiments/effect_estimation/effect_experiments.yaml --experiment ci_agentgraph

