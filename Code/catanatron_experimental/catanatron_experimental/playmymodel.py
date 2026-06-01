from catanatron.cli import register_cli_player
from .ppo_player import PPOPlayer

register_cli_player("PPO")(PPOPlayer)
