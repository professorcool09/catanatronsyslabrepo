# canatron-syslab Code Guide

Repository: https://github.com/Zawiop/canatron-syslab

## Purpose

This repository contains the main reinforcement-learning code for the CaRL Catan bot. It is based on Catanatron and adds project-specific training, reward functions, trained model files, and command-line evaluation scripts. Use this repository to understand how the agent was trained and how the trained policy is loaded for automated games.

## Main ideas

- The project uses Catanatron as the Catan simulator and Gymnasium environment.
- The learning algorithm is MaskablePPO from `sb3-contrib`.
- The model receives a 614-value Catan game-state vector.
- The policy network scores a 290-action discrete action space.
- Invalid Catan moves are removed with action masking before the model chooses an action.
- Reward shaping is used so the agent receives feedback for victory points, settlements, cities, wins, and losses.

## Files and folders to inspect first

| Path | What it is for |
| --- | --- |
| `TRAIN.py` | Main training entry point for MaskablePPO experiments. |
| `TRAINWITHMODELS.py` | Training setup for experiments involving stronger opponent bots. |
| `PLAYMYMODEL.py` | Loads a saved PPO model and registers it as a Catanatron CLI player. |
| `requirements.txt` | Python dependencies needed to run the project. |
| `checkpoints/` | Saved intermediate model checkpoints during training. |
| `FINALMODEL*`, `FourthReward/`, `LongTermReward/` | Saved models and experiment folders from different reward-function versions. |
| `catanatron/` | The Catanatron simulator code used by the project. |
| `ui/` and `Dockerfile.web` | Web/demo-related files inside the main project repository. |

## How to run locally

```bash
git clone https://github.com/Zawiop/canatron-syslab.git
cd canatron-syslab
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If editable installation is needed for Catanatron, run:

```bash
pip install -e .
pip install -e ".[gym]"
```

## Training overview

A typical training run creates the Catanatron Gymnasium environment, wraps it with an action masker, and trains a MaskablePPO model. The high-level flow is:

1. Start a Catan game simulation.
2. Convert the board and player state into numerical features.
3. Mask illegal actions.
4. Let MaskablePPO choose from the legal actions.
5. Apply the reward function.
6. Save checkpoints and the final model.

## Testing a trained model

The model can be tested through the Catanatron command line after the PPO player is registered in `PLAYMYMODEL.py`. The general format is:

```bash
catanatron-play --code=PLAYMYMODEL.py --num 1000 --players R,RL,R,R
```

Common opponent codes used in the project include random bots, weighted-random bots, MCTS, and value-function bots.

## What to look for when reading the code

Start with the reward function and masking logic. Those are the most important project-specific changes. Then follow how the observation is created, how the valid action mask is built, and how the model prediction is mapped back into a playable Catanatron action.

## Known limitations

- Longer training did not always improve performance monotonically.
- Strong opponent bots slow down training heavily.
- The model performs much better against random-like bots than against the strongest value-function bot.
- Trading with opponents is still limited compared with full human Catan.
