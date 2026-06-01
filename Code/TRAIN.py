import gymnasium as gym
import numpy as np
import torch
import catanatron.gym

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.models.player import RandomPlayer


# -------------------------
# MASK
# -------------------------
def mask_fn(env) -> np.ndarray:
    n = env.action_space.n
    valid = np.array(env.unwrapped.get_valid_actions(), dtype=np.int64).flatten()

    mask = np.zeros(n, dtype=np.bool_)
    valid = valid[(valid >= 0) & (valid < n)]

    if valid.size == 0:
        raise RuntimeError("Zero valid actions")

    mask[valid] = True
    return mask


def vp_reward(game, p0_color):
    winning_color = game.winning_color()
    if winning_color is None:
        return 0.0
    return 1.0 if winning_color == p0_color else -1.0

class RewardIteration2:
    def __init__(self):
        self.prev_vp = None
        self.prev_settlements = None
        self.prev_cities = None

    def _safe_get(self, obj, key, default=0):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _player_state(self, game, p0_color):
        ps = game.state.player_state
        if isinstance(ps, dict):
            if p0_color in ps:
                return ps[p0_color]
            for k, v in ps.items():
                if str(k) == str(p0_color):
                    return v
        return None

    def _count_structures(self, game, p0_color):
        settlements = 0
        cities = 0
        for _, b in game.state.board.buildings.items():
            color = self._safe_get(b, "color", None)
            btype = str(self._safe_get(b, "building_type", self._safe_get(b, "type", ""))).lower()
            if str(color) == str(p0_color):
                if btype.endswith("city"):
                    cities += 1
                else:
                    settlements += 1
        return settlements, cities

    def __call__(self, game, p0_color):
        winner = game.winning_color()
        if winner is not None:
            return 10.0 if str(winner) == str(p0_color) else -10.0

        player = self._player_state(game, p0_color)
        vp = self._safe_get(player, "victory_points",
             self._safe_get(player, "actual_victory_points",
             self._safe_get(player, "public_victory_points", 0)))

        settlements, cities = self._count_structures(game, p0_color)

        if self.prev_vp is None:
            self.prev_vp = vp
            self.prev_settlements = settlements
            self.prev_cities = cities
            return 0.0

        reward = 0.0
        reward += 2.5 * (vp - self.prev_vp)
        reward += 1.5 * (settlements - self.prev_settlements)
        reward += 2.5 * (cities - self.prev_cities)

        self.prev_vp = vp
        self.prev_settlements = settlements
        self.prev_cities = cities
        return float(reward)

import math

class ExpansionReward:
    def __init__(self):
        self.last_game_id = None
        self.prev_vp = None
        self.prev_settlements = None
        self.prev_cities = None
        self.prev_roads = None
        self.prev_resources = None
        self.prev_dev_cards = None

    def _safe_get(self, obj, key, default=0):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _player_state(self, game, p0_color):
        ps = game.state.player_state
        if isinstance(ps, dict):
            if p0_color in ps:
                return ps[p0_color]
            for k, v in ps.items():
                if str(k) == str(p0_color):
                    return v
        return None

    def _count_structures(self, game, p0_color):
        settlements = 0
        cities = 0

        for _, b in game.state.board.buildings.items():
            color = self._safe_get(b, "color", None)
            btype = str(
                self._safe_get(
                    b,
                    "building_type",
                    self._safe_get(b, "type", "")
                )
            ).lower()

            if str(color) == str(p0_color):
                if btype.endswith("city"):
                    cities += 1
                else:
                    settlements += 1

        return settlements, cities

    def _count_roads(self, game, p0_color):
        roads = 0

        road_data = getattr(game.state.board, "roads", {})

        if isinstance(road_data, dict):
            iterable = road_data.values()
        else:
            iterable = road_data

        for r in iterable:
            color = self._safe_get(r, "color", None)
            if str(color) == str(p0_color):
                roads += 1

        return roads

    def _count_resources(self, player):
        resources = self._safe_get(player, "resources", {})

        if isinstance(resources, dict):
            return sum(resources.values())

        try:
            return sum(resources)
        except Exception:
            return 0

    def _count_dev_cards(self, player):
        dev_cards = self._safe_get(
            player,
            "development_cards",
            self._safe_get(player, "dev_cards", {})
        )

        if isinstance(dev_cards, dict):
            return sum(dev_cards.values())

        try:
            return len(dev_cards)
        except Exception:
            return 0

    def __call__(self, game, p0_color):
        game_id = id(game)

        player = self._player_state(game, p0_color)
        if player is None:
            return 0.0

        vp = float(
            self._safe_get(
                player,
                "victory_points",
                self._safe_get(
                    player,
                    "actual_victory_points",
                    self._safe_get(player, "public_victory_points", 0),
                ),
            )
        )

        settlements, cities = self._count_structures(game, p0_color)
        roads = self._count_roads(game, p0_color)
        resources = self._count_resources(player)
        dev_cards = self._count_dev_cards(player)

        if self.last_game_id != game_id:
            self.last_game_id = game_id
            self.prev_vp = vp
            self.prev_settlements = settlements
            self.prev_cities = cities
            self.prev_roads = roads
            self.prev_resources = resources
            self.prev_dev_cards = dev_cards
            return 0.0

        reward = 0.0

        # Main progress
        reward += 2.5 * (vp - self.prev_vp)
        reward += 1.5 * (settlements - self.prev_settlements)
        reward += 2.5 * (cities - self.prev_cities)

        # Expansion / development shaping
        reward += 0.25 * (roads - self.prev_roads)
        reward += 0.15 * (resources - self.prev_resources)
        reward += 0.25 * (dev_cards - self.prev_dev_cards)

        # Prefer faster wins, but keep penalty tiny
        reward -= 0.005

        winner = game.winning_color()
        if winner is not None:
            reward += 20.0 if str(winner) == str(p0_color) else -20.0

        self.prev_vp = vp
        self.prev_settlements = settlements
        self.prev_cities = cities
        self.prev_roads = roads
        self.prev_resources = resources
        self.prev_dev_cards = dev_cards

        return float(reward)

# -------------------------
# SINGLE ENV CREATOR
# -------------------------
def make_env():
    reward = ExpansionReward()
    env = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "reward_function": reward,
            "invalid_action_reward": -1.0,
        },
    )
    env = ActionMasker(env, mask_fn)
    return env



if __name__ == "__main__":

    print("CUDA available:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    N_ENVS =8
    
    CONTINUE_FROM = "FINALMODEL/MYMODEL_final40000000.zip"
    CONTINUE_VECNORM = "FINALMODEL/vec_normalize.pkl"
    
    venv = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    
    if CONTINUE_FROM and CONTINUE_VECNORM:
        print("Loading VecNormalize from:", CONTINUE_VECNORM)
        venv = VecNormalize.load(CONTINUE_VECNORM, venv)
        venv.training = True
        venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
    if CONTINUE_FROM:
        print("Loading existing model from:", CONTINUE_FROM)
        model = MaskablePPO.load(
            CONTINUE_FROM,
            env=venv,
            device=device,
        )
        model.verbose = 1
    else:
        print("Starting new model")
        model = MaskablePPO(
            "MlpPolicy",
            venv,
            device=device,
            verbose=1,
            learning_rate=1e-4,
            ent_coef=0.05,
            n_steps=1024,
            batch_size=512,
            gamma=0.99,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=1000000,
        save_path="./checkpoints/",
        name_prefix="catan",
    )
    hours = 8
    timesteps = 5000000*hours
    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint_cb)
    finally:
        model.save(f"VALUE_TRAINED_MODEL_final{timesteps}")
        venv.save("value_trained_vec_normalize.pkl")
        print("Training finished + saved.")
 