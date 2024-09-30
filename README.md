
# Overview
This is the accompanying repository to the publication "Approximating Energy 
Market Clearing and Bidding With Model-Based Reinforcement Learning" (TODO: Add URL after publication)

# Installation
Run `pip install -r requirements.txt` for installation. Tested with python 3.8. 
If automatic torch installation does not work, try [manual installation of torch](https://pytorch.org/get-started/locally/) first.  

Note that this repository contains outdated versions of the drl repo for and mlopf repository. 
The up-to-date repositories can be found at https://gitlab.com/thomaswolgast/drl and https://gitlab.com/thomaswolgast/mlopf. 

Note that the focus of the `mlopf` library shifted from multi-agent bidding to single-agent OPF approximation environments. 
Details can be found in the publication "[Learning the optimal power flow: Environment design matters](https://www.sciencedirect.com/science/article/pii/S2666546824000764)"

# Run experiments (minimal examples)
After installation, run in the `src` directory:  
`python main.py --agent "general_market_maddpg:MarketMaddpgPab" --environ "ml_opf.envs.energy_market_bidding:OpfAndBiddingEcoDispatchEnv" --hyper "{'start_train': 300, 'start_train_agents': 400}" --env-hyperparams "{'n_agents': 10}" --steps 500 --store --num 1 --test-steps 3`
for the model-based MADDPG experiments or
`python main.py --agent "maddpg:MarketMaddpg" --environ "ml_opf.envs.energy_market_bidding:BiddingEcoDispatchEnv" --hyper "{'start_train': 300}" --env-hyperparams "{'n_agents': 10}" --steps 400 --store --num 1`
for the baseline experiments. 