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
from catanatron.players.weighted_random import WeightedRandomPlayer



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

class RewardIteration3:
    """
    Board-development reward with:
    - Strong city/settlement incentives
    - Light dev-card penalty
    - Starting-placement bonus for good dice/resource diversity
    - Robber-placement penalty if robber is placed on your own settlement/city
      or on an empty/unowned area
    """

    def __init__(self):
        self.last_game_id = None

        self.prev_vp = None
        self.prev_settlements = None
        self.prev_cities = None
        self.prev_roads = None
        self.prev_dev_cards = None
        self.prev_robber_coord = None
        self.prev_own_nodes = set()

        # 2/12 low, 6/8 high
        self.dice_pips = {
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            8: 5,
            9: 4,
            10: 3,
            11: 2,
            12: 1,
        }

        # Brick/lumber/wheat are weighted strongly for expansion.
        # Ore is useful, but we keep it lower because it mostly supports cities/dev cards.
        self.resource_weights = {
            "brick": 1.45,
            "wood": 1.30,
            "lumber": 1.30,
            "grain": 1.20,
            "wheat": 1.20,
            "sheep": 0.95,
            "wool": 0.95,
            "ore": 0.75,
            "desert": 0.0,
            "none": 0.0,
        }

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

    def _building_color(self, building):
        return self._safe_get(building, "color", None)

    def _building_type(self, building):
        return str(
            self._safe_get(
                building,
                "building_type",
                self._safe_get(building, "type", "")
            )
        ).lower()

    def _count_structures(self, game, p0_color):
        settlements = 0
        cities = 0

        for _, building in game.state.board.buildings.items():
            color = self._building_color(building)
            btype = self._building_type(building)

            if str(color) == str(p0_color):
                if btype.endswith("city"):
                    cities += 1
                else:
                    settlements += 1

        return settlements, cities

    def _own_building_nodes(self, game, p0_color):
        nodes = set()

        for node, building in game.state.board.buildings.items():
            color = self._building_color(building)

            if str(color) == str(p0_color):
                nodes.add(node)

        return nodes

    def _count_roads(self, game, p0_color):
        roads = 0
        road_data = getattr(game.state.board, "roads", {})

        if isinstance(road_data, dict):
            iterable = road_data.values()
        else:
            iterable = road_data

        for road in iterable:
            color = self._safe_get(road, "color", None)

            if str(color) == str(p0_color):
                roads += 1

        return roads

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

    # -------------------------
    # TILE / BOARD HELPERS
    # -------------------------

    def _iter_tiles(self, game):
        board = game.state.board
        board_map = getattr(board, "map", None)

        candidates = [
            getattr(board, "tiles", None),
            getattr(board, "land_tiles", None),
            getattr(board_map, "tiles", None) if board_map is not None else None,
            getattr(board_map, "land_tiles", None) if board_map is not None else None,
        ]

        for candidate in candidates:
            if candidate is None:
                continue

            if isinstance(candidate, dict):
                return list(candidate.items())

            try:
                return [(self._tile_coord(tile, None), tile) for tile in candidate]
            except Exception:
                pass

        return []

    def _tile_coord(self, tile, fallback):
        for attr in ["coordinate", "coords", "coord", "id"]:
            value = self._safe_get(tile, attr, None)
            if value is not None:
                return value
        return fallback

    def _tile_number(self, tile):
        for attr in ["number", "dice_number", "roll", "production_number"]:
            value = self._safe_get(tile, attr, None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    return None
        return None

    def _tile_resource(self, tile):
        for attr in ["resource", "resource_type", "tile_type"]:
            value = self._safe_get(tile, attr, None)
            if value is not None:
                return str(value).lower()
        return "none"

    def _adjacent_nodes_for_tile(self, game, tile_coord, tile):
        # Try direct tile attributes first
        for attr in ["nodes", "corners", "intersections", "vertices"]:
            value = self._safe_get(tile, attr, None)
            if value is not None:
                try:
                    return list(value)
                except Exception:
                    pass

        # Try board/map helper methods
        board = game.state.board
        board_map = getattr(board, "map", None)

        for obj in [board, board_map]:
            if obj is None:
                continue

            for method_name in [
                "adjacent_nodes",
                "get_adjacent_nodes",
                "get_nodes",
                "nodes_for_tile",
                "tile_nodes",
            ]:
                method = getattr(obj, method_name, None)
                if method is not None:
                    try:
                        return list(method(tile_coord))
                    except Exception:
                        pass

        return []

    def _adjacent_tiles_for_node(self, game, node):
        adjacent = []

        for tile_coord, tile in self._iter_tiles(game):
            nodes = self._adjacent_nodes_for_tile(game, tile_coord, tile)

            if node in nodes:
                adjacent.append((tile_coord, tile))

        return adjacent

    # -------------------------
    # STARTING PLACEMENT BONUS
    # -------------------------

    def _starting_node_score(self, game, node):
        adjacent_tiles = self._adjacent_tiles_for_node(game, node)

        resources_seen = set()
        production_score = 0.0

        for _, tile in adjacent_tiles:
            number = self._tile_number(tile)
            resource = self._tile_resource(tile)

            if resource in ["desert", "none"]:
                continue

            resources_seen.add(resource)

            pips = self.dice_pips.get(number, 0)
            resource_weight = self.resource_weights.get(resource, 0.75)

            production_score += pips * resource_weight

        # Scale down so this does not dominate the whole reward.
        production_score *= 0.20

        diversity_bonus = 0.40 * len(resources_seen)

        # Extra early-game resource priorities.
        essential_bonus = 0.0
        if "brick" in resources_seen:
            essential_bonus += 0.75
        if "wood" in resources_seen or "lumber" in resources_seen:
            essential_bonus += 0.65
        if "wheat" in resources_seen or "grain" in resources_seen:
            essential_bonus += 0.55
        if "sheep" in resources_seen or "wool" in resources_seen:
            essential_bonus += 0.25
        if "ore" in resources_seen:
            essential_bonus += 0.10

        return production_score + diversity_bonus + essential_bonus

    def _starting_placement_reward(self, game, p0_color, settlements, delta_settlements):
        if delta_settlements <= 0:
            return 0.0

        # Only reward first two settlement placements.
        if settlements > 2:
            return 0.0

        current_nodes = self._own_building_nodes(game, p0_color)
        new_nodes = current_nodes - self.prev_own_nodes

        reward = 0.0

        for node in new_nodes:
            reward += self._starting_node_score(game, node)

        return reward

    # -------------------------
    # ROBBER PENALTY / REWARD
    # -------------------------

    def _robber_coord(self, game):
        board = game.state.board

        for attr in [
            "robber_coordinate",
            "robber_coord",
            "robber_hex",
            "robber_tile",
            "robber",
        ]:
            value = self._safe_get(board, attr, None)
            if value is not None:
                return value

        for attr in [
            "robber_coordinate",
            "robber_coord",
            "robber_hex",
            "robber_tile",
            "robber",
        ]:
            value = self._safe_get(game.state, attr, None)
            if value is not None:
                return value

        return None

    def _robber_reward(self, game, p0_color):
        current_robber = self._robber_coord(game)

        if current_robber is None:
            return 0.0

        if self.prev_robber_coord is None:
            return 0.0

        # Only score robber when it moves.
        if current_robber == self.prev_robber_coord:
            return 0.0

        tile = None
        for tile_coord, candidate_tile in self._iter_tiles(game):
            if tile_coord == current_robber:
                tile = candidate_tile
                break

        if tile is None:
            return 0.0

        adjacent_nodes = self._adjacent_nodes_for_tile(game, current_robber, tile)

        own_adjacent = 0
        enemy_adjacent = 0

        for node in adjacent_nodes:
            building = game.state.board.buildings.get(node, None)
            if building is None:
                continue

            color = self._building_color(building)

            if str(color) == str(p0_color):
                own_adjacent += 1
            else:
                enemy_adjacent += 1

        reward = 0.0

        # Bad robber placement: blocks yourself.
        if own_adjacent > 0:
            reward -= 2.0 * own_adjacent

        # Bad robber placement: nobody nearby, so it blocks no one.
        if own_adjacent == 0 and enemy_adjacent == 0:
            reward -= 1.0

        # Good robber placement: blocks opponents.
        if enemy_adjacent > 0 and own_adjacent == 0:
            reward += 0.75 * enemy_adjacent

        return reward

    # -------------------------
    # MAIN REWARD
    # -------------------------

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
        dev_cards = self._count_dev_cards(player)
        current_robber = self._robber_coord(game)
        current_own_nodes = self._own_building_nodes(game, p0_color)

        if self.last_game_id != game_id:
            self.last_game_id = game_id
            self.prev_vp = vp
            self.prev_settlements = settlements
            self.prev_cities = cities
            self.prev_roads = roads
            self.prev_dev_cards = dev_cards
            self.prev_robber_coord = current_robber
            self.prev_own_nodes = current_own_nodes
            return 0.0

        delta_vp = vp - self.prev_vp
        delta_settlements = settlements - self.prev_settlements
        delta_cities = cities - self.prev_cities
        delta_roads = roads - self.prev_roads
        delta_dev_cards = dev_cards - self.prev_dev_cards

        reward = 0.0

        # Base reward, close to RewardIteration2 but with slightly stronger city bias.
        reward += 2.0 * delta_vp
        reward += 1.5 * delta_settlements
        reward += 4.0 * delta_cities
        reward += 0.10 * delta_roads

        # Very light dev-card penalty. Do not make dev cards toxic.
        reward -= 0.02 * max(delta_dev_cards, 0)

        # Reward strong initial settlement placement.
        reward += self._starting_placement_reward(
            game,
            p0_color,
            settlements,
            delta_settlements,
        )

        # Penalize bad robber placements and reward blocking enemies.
        reward += self._robber_reward(game, p0_color)

        # Small urgency penalty.
        reward -= 0.005

        winner = game.winning_color()
        if winner is not None:
            reward += 25.0 if str(winner) == str(p0_color) else -25.0

        self.prev_vp = vp
        self.prev_settlements = settlements
        self.prev_cities = cities
        self.prev_roads = roads
        self.prev_dev_cards = dev_cards
        self.prev_robber_coord = current_robber
        self.prev_own_nodes = current_own_nodes

        return float(reward)
# -------------------------
# SINGLE ENV CREATOR
# -------------------------
def make_env():
    reward = RewardIteration3()

    env = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "reward_function": reward,
            "invalid_action_reward": -1.0,
            "enemies": [
                ValueFunctionPlayer(Color.RED),
                WeightedRandomPlayer(Color.ORANGE),
                WeightedRandomPlayer(Color.WHITE),
            ],
        },
    )

    env = ActionMasker(env, mask_fn)
    return env



if __name__ == "__main__":

    print("CUDA available:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    N_ENVS =12
    
    CONTINUE_FROM = "FINALMODEL4PLAYERS/RRR15MR10"
    CONTINUE_VECNORM = "FINALMODEL4PLAYERS/RRR15MR10.pkl"
    
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
            learning_rate=1e-5,
            ent_coef=0.005,
        )
        model.verbose = 1
    else:
        print("Starting new model")
        model = MaskablePPO(
            "MlpPolicy",
            venv,
            device=device,
            verbose=1,
            learning_rate=5e-5,
            ent_coef=0.02,
            n_steps=2048,
            batch_size=1024,
            gamma=0.99,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=1000000,
        save_path="./checkpoints/",
        name_prefix="catan",
    )
    ms = 1
    timesteps = 1000000*ms
    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint_cb)
    finally:
        model.save("RRR16MR10")
        venv.save("RRR16MR10.pkl")
        print("Training finished + saved.")
 