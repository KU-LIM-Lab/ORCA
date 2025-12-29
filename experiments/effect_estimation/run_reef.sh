# Run experiments for effect estimation
# idhp oracle, ci oracle, ci agentgraph
# how to run: bash experiments/effect_estimation/run.sh

python -m experiments.effect_estimation.run_experiment --config experiments/effect_estimation/effect_experiments.yaml --experiment reef_full_pipeline

# python -m experiments.effect_estimation.generate_summary --results-dir experiments/results/effect_estimation --setting full_pipeline --dataset reef