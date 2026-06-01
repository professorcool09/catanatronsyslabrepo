import os
import numpy as np

from catanatron import Player
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.catanatron_env import to_action_space
from sb3_contrib.ppo_mask import MaskablePPO

FEATURES = get_feature_ordering(num_players=2)

class PPOPlayer(Player):
    def __init__(self, color, model_path=None):
        super().__init__(color)
        model_path = model_path or os.getenv("CATAN_PPO_MODEL", "/app/MYMODEL_final20000000.zip")
        print("✅ Loading PPO model:", model_path, flush=True)
        self.model = MaskablePPO.load(model_path)

    def decide(self, game, playable_actions):
        sample = create_sample(game, self.color)
        obs = np.array([float(sample[f]) for f in FEATURES], dtype=np.float32)

        n = int(self.model.action_space.n)
        mask = np.zeros(n, dtype=bool)
        idx_to_action = {}
        for a in playable_actions:
            idx = to_action_space(a)
            if 0 <= idx < n:
                mask[idx] = True
                idx_to_action[idx] = a

        if not idx_to_action:
            return playable_actions[0]

        action_idx, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return idx_to_action.get(int(action_idx), playable_actions[0])
