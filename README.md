Scripts for distributional RL experiments involving Atari ALE. Codebase modified from https://github.com/Kaixhin/Rainbow with additional agents, architecture models, and evaluation procedures. Project still under development.

**Modifications from original include:**

main.py -- support for checkpointing to allow resumption of interrupted training rather than only retaining weights of best performing agent. Support for storing results from individual random seeds.

main_store_frame.py -- script for capturing game frames as agent reaches certain score thresholds in Breakout. Used for evaluation of trained agent behaviour posthoc.

agent.py and model.py -- additional agents, including Rainbow agent using non-distributional DQN rather than C51 output, and agents that learn return variance as an auxiliary loss through the TD-algorithm of Sherstan et al (2018), and a DQN agent that outputs a vector of Q-value estimates with different fixed biases.
