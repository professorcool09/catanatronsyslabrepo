# ppo_player.py
import os
import pickle
import numpy as np

from catanatron import Player
from catanatron.cli import register_cli_player
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.catanatron_env import to_action_space

from sb3_contrib.ppo_mask import MaskablePPO


# IMPORTANT: 4-player model needs 4-player feature ordering
FEATURES = get_feature_ordering(num_players=4)


class RLPlayer(Player):
    def __init__(self, color, model_path=None):
        super().__init__(color)

        model_path = model_path or os.getenv(
            "CATAN_PPO_MODEL",
            "FINALMODEL4PLAYERS/FFF34M"
        )

        vecnorm_path = os.getenv(
            "CATAN_VECNORM",
            "FINALMODEL4PLAYERS/FFF34.pkl"
        )

        self.model = MaskablePPO.load(model_path)

        with open(vecnorm_path, "rb") as f:
            self.vecnorm = pickle.load(f)

        self.vecnorm.training = False
        self.vecnorm.norm_reward = False

    def decide(self, game, playable_actions):
        sample = create_sample(game, self.color)

        try:
            obs = np.array([float(sample[f]) for f in FEATURES], dtype=np.float32)
        except KeyError as e:
            print("Feature mismatch.")
            print("Missing feature:", e)
            print("Expected feature count:", len(FEATURES))
            print("Sample feature count:", len(sample))
            return playable_actions[0]

        obs = obs.reshape(1, -1)
        obs = self.vecnorm.normalize_obs(obs)

        n = int(self.model.action_space.n)
        mask = np.zeros(n, dtype=bool)
        idx_to_action = {}

        for a in playable_actions:
            try:
                idx = to_action_space(a)
                if 0 <= idx < n:
                    mask[idx] = True
                    idx_to_action[idx] = a
            except Exception:
                pass

        if not idx_to_action:
            return playable_actions[0]

        if len(idx_to_action) == 1:
            return next(iter(idx_to_action.values()))

        mask = mask.reshape(1, -1)

        try:
            action_idx, _ = self.model.predict(
                obs,
                action_masks=mask,
                deterministic=True
            )
        except ValueError as e:
            print("PPO predict crashed.")
            print("obs shape:", obs.shape)
            print("obs min/max:", obs.min(), obs.max())
            print("mask shape:", mask.shape)
            print("valid count:", mask.sum())
            print("valid indexes:", np.where(mask[0])[0])
            print(e)
            return playable_actions[0]

        action_idx = int(np.asarray(action_idx).item())

        return idx_to_action.get(action_idx, playable_actions[0])


register_cli_player("RL", RLPlayer)