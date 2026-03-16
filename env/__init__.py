import gymnasium

gymnasium.register(
    id="Snake-v0",
    entry_point="env.snake_env:SnakeEnv",
)

from env.snake_env import SnakeEnv  # noqa: E402, F401
